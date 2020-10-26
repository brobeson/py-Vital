"""A GOT10K tracker that implements Vital."""

import time

import got10k.trackers
import numpy
import torch
import yaml

import gnet.g_init
import gnet.g_pretrain
import modules.model
import modules.sample_generator
import modules.utils
import tracking.data_prov
import tracking.bbreg
import tracking.gen_config

opts = yaml.safe_load(open("./tracking/options.yaml", "r"))


def _fix_positive_samples(samples: numpy.ndarray, n: int, box: numpy.ndarray) -> numpy.ndarray:
    """
    Ensure a set of positive samples is valid.

    :param numpy.ndarray samples: The positive samples to fix if invalid.
    :param int n: The number of samples to create if fixing is needed.
    :param numpy.ndarray box: The target bounding box used if fixing is needed.
    :return: The original samples if no fixing is required, or the fixed samples if fixing is
        required.
    :rtype: np.ndarray
    """
    if samples.shape[0] > 0:
        return samples
    return numpy.tile(box, [n, 1])


def forward_samples(model, image, samples, out_layer="conv3"):
    """Forward samples through the network."""
    model.eval()
    extractor = tracking.data_prov.RegionExtractor(image, samples, opts)
    for i, regions in enumerate(extractor):
        if opts["use_gpu"]:
            regions = regions.cuda()
        with torch.no_grad():
            feat = model(regions, out_layer=out_layer)
        if i == 0:
            feats = feat.detach().clone()
        else:
            feats = torch.cat((feats, feat.detach().clone()), 0)
    return feats


def train(
    model, model_g, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer="fc4"
):
    """Train the models."""
    model.train()

    batch_pos = opts["batch_pos"]
    batch_neg = opts["batch_neg"]
    batch_test = opts["batch_test"]
    batch_neg_cand = max(opts["batch_neg_cand"], batch_neg)

    pos_idx = numpy.random.permutation(pos_feats.size(0))
    neg_idx = numpy.random.permutation(neg_feats.size(0))
    while len(pos_idx) < batch_pos * maxiter:
        pos_idx = numpy.concatenate(
            [pos_idx, numpy.random.permutation(pos_feats.size(0))]
        )
    while len(neg_idx) < batch_neg_cand * maxiter:
        neg_idx = numpy.concatenate(
            [neg_idx, numpy.random.permutation(neg_feats.size(0))]
        )
    pos_pointer = 0
    neg_pointer = 0

    for _ in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = pos_feats[pos_cur_idx]
        if model_g is not None:
            batch_asdn_feats = pos_feats.index_select(0, pos_cur_idx)
        batch_neg_feats = neg_feats[neg_cur_idx]

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                with torch.no_grad():
                    score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start == 0:
                    neg_cand_score = score.detach()[:, 1].clone()
                else:
                    neg_cand_score = torch.cat(
                        (neg_cand_score, score.detach()[:, 1].clone()), 0
                    )

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats[top_idx]
            model.train()

        if model_g is not None:
            model_g.eval()
            res_asdn = model_g(batch_asdn_feats)
            model_g.train()
            num = res_asdn.size(0)
            mask_asdn = torch.ones(num, 512, 3, 3)
            res_asdn = res_asdn.view(num, 3, 3)
            for j in range(num):
                feat_ = res_asdn[j, :, :]
                featlist = feat_.view(1, 9).squeeze()
                feat_list = featlist.detach().cpu().numpy()
                idlist = feat_list.argsort()
                idxlist = idlist[:3]

                for k, idx in enumerate(idxlist):
                    row = idx // 3
                    col = idx % 3
                    mask_asdn[:, :, col, row] = 0
            mask_asdn = mask_asdn.view(mask_asdn.size(0), -1)
            if opts["use_gpu"]:
                batch_asdn_feats = batch_asdn_feats.cuda()
                mask_asdn = mask_asdn.cuda()
            batch_asdn_feats = batch_asdn_feats * mask_asdn

        # forward
        if model_g is None:
            pos_score = model(batch_pos_feats, in_layer=in_layer)
        else:
            pos_score = model(batch_asdn_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        if "grad_clip" in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts["grad_clip"])
        optimizer.step()

        if model_g is not None:
            start = time.time()
            prob_k = torch.zeros(9)
            for k in range(9):
                row = k // 3
                col = k % 3

                model.eval()
                batch = batch_pos_feats.view(batch_pos, 512, 3, 3)
                batch[:, :, col, row] = 0
                batch = batch.view(batch.size(0), -1)

                if opts["use_gpu"]:
                    batch = batch.cuda()

                prob = model(batch, in_layer="fc4", out_layer="fc6_softmax")[:, 1]
                model.train()

                prob_k[k] = prob.sum()

            _, idx = torch.min(prob_k, 0)
            idx = idx.item()
            row = idx // 3
            col = idx % 3

            optimizer_g = gnet.g_init.set_optimizer_g(model_g)
            labels = torch.ones(batch_pos, 1, 3, 3)
            labels[:, :, col, row] = 0

            batch_pos_feats = batch_pos_feats.view(batch_pos_feats.size(0), -1)
            res = model_g(batch_pos_feats)
            labels = labels.view(batch_pos, -1)
            criterion_g = torch.nn.MSELoss(reduction="mean")
            loss_g_2 = criterion_g(res.float(), labels.cuda().float())
            model_g.zero_grad()
            loss_g_2.backward()
            optimizer_g.step()

            end = time.time()
            # print("asdn objective %.3f, %.2f s" % (loss_g_2, end - start))


class Vital(got10k.trackers.Tracker):
    """A GOT10K tracker that implements Vital."""

    def __init__(self, name: str):
        super().__init__(name=name, is_deterministic=False)
        self.target_bbox = None
        self.model = None
        self.model_g = None
        self.criterion = None
        self.criterion_g = None
        self.update_optimizer = None
        self.bbreg = None
        self.sample_generator = None
        self.pos_generator = None
        self.neg_generator = None
        self.pos_feats_all = None
        self.neg_feats_all = None
        self.current_frame = 0

    def init(self, image, box):
        self.target_bbox = numpy.array(box)

        # Initialize the model.
        self.model = modules.model.MDNet(opts["model_path"])
        self.model_g = gnet.g_init.NetG()
        if opts["use_gpu"]:
            self.model = self.model.cuda()
            self.model_g = self.model_g.cuda()

        # Initialize the criterions and optimizers.
        self.criterion = modules.model.BCELoss()
        self.criterion_g = torch.nn.MSELoss(reduction="mean")
        self.model.set_learnable_params(opts["ft_layers"])
        self.model_g.set_learnable_params(opts["ft_layers"])
        init_optimizer = modules.model.set_optimizer(
            self.model, opts["lr_init"], opts["lr_mult"]
        )
        self.update_optimizer = modules.model.set_optimizer(
            self.model, opts["lr_update"], opts["lr_mult"]
        )

        # Draw positive and negative samples.
        pos_examples = _fix_positive_samples(
            modules.sample_generator.SampleGenerator(
                "gaussian", image.size, opts["trans_pos"], opts["scale_pos"]
            )(box, opts["n_pos_init"], opts["overlap_pos_init"]),
            opts["n_pos_init"],
            box,
        )

        neg_examples = numpy.concatenate(
            [
                modules.sample_generator.SampleGenerator(
                    "uniform",
                    image.size,
                    opts["trans_neg_init"],
                    opts["scale_neg_init"],
                )(box, int(opts["n_neg_init"] * 0.5), opts["overlap_neg_init"]),
                modules.sample_generator.SampleGenerator("whole", image.size)(
                    box, int(opts["n_neg_init"] * 0.5), opts["overlap_neg_init"]
                ),
            ]
        )
        neg_examples = numpy.random.permutation(neg_examples)

        # Extract positive and negative features.
        pos_feats = forward_samples(self.model, image, pos_examples)
        neg_feats = forward_samples(self.model, image, neg_examples)

        # Initial training
        train(
            self.model,
            None,
            self.criterion,
            init_optimizer,
            pos_feats,
            neg_feats,
            opts["maxiter_init"],
        )
        del init_optimizer, neg_feats
        torch.cuda.empty_cache()
        gnet.g_pretrain.g_pretrain(
            self.model, self.model_g, self.criterion_g, pos_feats
        )
        torch.cuda.empty_cache()

        # Train bbox regressor
        bbreg_examples = _fix_positive_samples(
            modules.sample_generator.SampleGenerator(
                "uniform",
                image.size,
                opts["trans_bbreg"],
                opts["scale_bbreg"],
                opts["aspect_bbreg"],
            )(box, opts["n_bbreg"], opts["overlap_bbreg"]),
            opts["n_bbreg"],
            box,
        )
        bbreg_feats = forward_samples(self.model, image, bbreg_examples)
        self.bbreg = tracking.bbreg.BBRegressor(image.size)
        self.bbreg.train(bbreg_feats, bbreg_examples, box)
        del bbreg_feats
        torch.cuda.empty_cache()

        # Init sample generators for update
        self.sample_generator = modules.sample_generator.SampleGenerator(
            "gaussian", image.size, opts["trans"], opts["scale"]
        )
        self.pos_generator = modules.sample_generator.SampleGenerator(
            "gaussian", image.size, opts["trans_pos"], opts["scale_pos"]
        )
        self.neg_generator = modules.sample_generator.SampleGenerator(
            "uniform", image.size, opts["trans_neg"], opts["scale_neg"]
        )

        # Init pos/neg features for update
        neg_examples = self.neg_generator(
            box, opts["n_neg_update"], opts["overlap_neg_init"]
        )
        neg_feats = forward_samples(self.model, image, neg_examples)
        self.pos_feats_all = [pos_feats]
        self.neg_feats_all = [neg_feats]
        self.current_frame = 0

    def update(self, image):
        self.current_frame += 1
        # Estimate target bbox
        samples = self.sample_generator(self.target_bbox, opts["n_samples"])
        sample_scores = forward_samples(self.model, image, samples, out_layer="fc6")

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
        success = target_score > 0

        # Expand search area at failure
        if success:
            self.sample_generator.set_trans(opts["trans"])
        else:
            self.sample_generator.expand_trans(opts["trans_limit"])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None, :]
            bbreg_feats = forward_samples(self.model, image, bbreg_samples)
            bbreg_samples = self.bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        self.target_bbox = bbreg_bbox

        # Data collect
        if success:
            pos_examples = _fix_positive_samples(
                self.pos_generator(
                    target_bbox, opts["n_pos_update"], opts["overlap_pos_update"]
                ),
                opts["n_pos_update"],
                target_bbox,
            )
            pos_feats = forward_samples(self.model, image, pos_examples)
            self.pos_feats_all.append(pos_feats)
            if len(self.pos_feats_all) > opts["n_frames_long"]:
                del self.pos_feats_all[0]

            neg_examples = self.neg_generator(
                target_bbox, opts["n_neg_update"], opts["overlap_neg_update"]
            )
            neg_feats = forward_samples(self.model, image, neg_examples)
            self.neg_feats_all.append(neg_feats)
            if len(self.neg_feats_all) > opts["n_frames_short"]:
                del self.neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts["n_frames_short"], len(self.pos_feats_all))
            pos_data = torch.cat(self.pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(self.neg_feats_all, 0)
            train(
                self.model,
                None,
                self.criterion,
                self.update_optimizer,
                pos_data,
                neg_data,
                opts["maxiter_update"],
            )

        # Long term update
        elif self.current_frame % opts["long_interval"] == 0:
            pos_data = torch.cat(self.pos_feats_all, 0)
            neg_data = torch.cat(self.neg_feats_all, 0)
            train(
                self.model,
                self.model_g,
                self.criterion,
                self.update_optimizer,
                pos_data,
                neg_data,
                opts["maxiter_update"],
            )

        torch.cuda.empty_cache()
        return self.target_bbox
