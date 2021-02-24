"""The VITAL tracker."""

import numpy
import PIL.Image
import torch
import yaml
from gnet.g_init import NetG, set_optimizer_g
from gnet.g_pretrain import g_pretrain
from modules.model import MDNet, BCELoss, set_optimizer
from modules.sample_generator import SampleGenerator
from tracking.data_prov import RegionExtractor
from tracking.bbreg import BBRegressor
import tracking.lr_schedules


def set_random_seeds(seed: int) -> None:
    """Seed the Numpy and PyTorch random generators."""
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def load_configuration(file_path: str) -> dict:
    """
    Load the VITAL runtime configuration from a file.

    :param str file_path: The path to the configuration file.
    :return: The loaded configuration.
    :rtype: dict
    :raises: FileNotFoundError if there is not file at `file_path`.
    """
    with open(file_path, "r") as configuration_file:
        return yaml.safe_load(configuration_file)


def forward_samples(model, image, samples, opts, out_layer="conv3"):
    model.eval()
    extractor = RegionExtractor(image, samples, opts)
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
    model,
    model_g,
    criterion,
    optimizer,
    pos_feats,
    neg_feats,
    maxiter,
    opts,
    in_layer="fc4",
):
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
            for i in range(num):
                feat_ = res_asdn[i, :, :]
                featlist = feat_.view(1, 9).squeeze()
                feat_list = featlist.detach().cpu().numpy()
                idlist = feat_list.argsort()
                idxlist = idlist[:3]

                for k in range(len(idxlist)):
                    idx = idxlist[k]
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
            # start = time.time()
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

            optimizer_g = set_optimizer_g(model_g)
            scheduler = tracking.lr_schedules.make_schedule(
                optimizer_g, opts["direction"], opts,
            )
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
            scheduler.step()


class VitalTracker:
    """The VITAL tracker."""

    def __init__(self, configuration: dict):
        self.custom_configuration = configuration
        self.opts = None
        self.update_optimizer = None
        self.pos_feats_all = None
        self.neg_feats_all = None
        self.sample_generator = None
        self.pos_generator = None
        self.target_bbox = None
        self.model = None
        self.model_g = None
        self.bbreg = None
        self.neg_generator = None
        self.criterion = None
        self.current_frame = None

    def initialize(self, init_bbox, image: PIL.Image.Image):
        """Initialize the tracker with the ground truth image and bounding box."""
        with open("tracking/options.yaml", "r") as yaml_file:
            self.opts = yaml.safe_load(yaml_file)
        self.opts.update(self.custom_configuration)

        self.target_bbox = numpy.array(init_bbox)
        self.current_frame = 0

        # Init model
        self.model = MDNet(self.opts["model_path"])
        self.model_g = NetG()
        if self.opts["use_gpu"]:
            self.model = self.model.cuda()
            self.model_g = self.model_g.cuda()

        # Init criterion and optimizer
        self.criterion = BCELoss()
        criterion_g = torch.nn.MSELoss(reduction="mean")
        self.model.set_learnable_params(self.opts["ft_layers"])
        self.model_g.set_learnable_params(self.opts["ft_layers"])
        init_optimizer = set_optimizer(
            self.model, self.opts["lr_init"], self.opts["lr_mult"]
        )
        self.update_optimizer = set_optimizer(
            self.model, self.opts["lr_update"], self.opts["lr_mult"]
        )

        # Draw pos/neg samples
        pos_examples = SampleGenerator(
            "gaussian", image.size, self.opts["trans_pos"], self.opts["scale_pos"]
        )(self.target_bbox, self.opts["n_pos_init"], self.opts["overlap_pos_init"])

        neg_examples = numpy.concatenate(
            [
                SampleGenerator(
                    "uniform",
                    image.size,
                    self.opts["trans_neg_init"],
                    self.opts["scale_neg_init"],
                )(
                    self.target_bbox,
                    int(self.opts["n_neg_init"] * 0.5),
                    self.opts["overlap_neg_init"],
                ),
                SampleGenerator("whole", image.size)(
                    self.target_bbox,
                    int(self.opts["n_neg_init"] * 0.5),
                    self.opts["overlap_neg_init"],
                ),
            ]
        )
        neg_examples = numpy.random.permutation(neg_examples)

        # Extract pos/neg features
        pos_feats = forward_samples(self.model, image, pos_examples, self.opts)
        neg_feats = forward_samples(self.model, image, neg_examples, self.opts)

        # Initial training
        train(
            self.model,
            None,
            self.criterion,
            init_optimizer,
            pos_feats,
            neg_feats,
            self.opts["maxiter_init"],
            self.opts,
        )
        del init_optimizer, neg_feats
        torch.cuda.empty_cache()
        g_pretrain(self.model, self.model_g, criterion_g, pos_feats, self.opts)
        torch.cuda.empty_cache()

        # Train bbox regressor
        bbreg_examples = SampleGenerator(
            "uniform",
            image.size,
            self.opts["trans_bbreg"],
            self.opts["scale_bbreg"],
            self.opts["aspect_bbreg"],
        )(self.target_bbox, self.opts["n_bbreg"], self.opts["overlap_bbreg"])
        bbreg_feats = forward_samples(self.model, image, bbreg_examples, self.opts)
        self.bbreg = BBRegressor(image.size)
        self.bbreg.train(bbreg_feats, bbreg_examples, self.target_bbox)
        del bbreg_feats
        torch.cuda.empty_cache()

        # Init sample generators for update
        self.sample_generator = SampleGenerator(
            "gaussian", image.size, self.opts["trans"], self.opts["scale"]
        )
        self.pos_generator = SampleGenerator(
            "gaussian", image.size, self.opts["trans_pos"], self.opts["scale_pos"]
        )
        self.neg_generator = SampleGenerator(
            "uniform", image.size, self.opts["trans_neg"], self.opts["scale_neg"]
        )

        # Init pos/neg features for update
        neg_examples = self.neg_generator(
            self.target_bbox, self.opts["n_neg_update"], self.opts["overlap_neg_init"]
        )
        neg_feats = forward_samples(self.model, image, neg_examples, self.opts)
        self.pos_feats_all = [pos_feats]
        self.neg_feats_all = [neg_feats]

    def find_target(self, image: PIL.Image.Image):
        """Find the target object in a frame."""
        self.current_frame += 1

        # Estimate target bbox
        samples = self.sample_generator(self.target_bbox, self.opts["n_samples"])
        sample_scores = forward_samples(
            self.model, image, samples, self.opts, out_layer="fc6"
        )
        if torch.any(torch.isnan(sample_scores)):
            raise RuntimeError("MDNet calculated NaN for scores.")

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        self.target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            self.target_bbox = self.target_bbox.mean(axis=0)
        success = target_score > 0

        # Expand search area at failure
        if success:
            self.sample_generator.set_trans(self.opts["trans"])
        else:
            self.sample_generator.expand_trans(self.opts["trans_limit"])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None, :]
            bbreg_feats = forward_samples(self.model, image, bbreg_samples, self.opts)
            bbreg_samples = self.bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = self.target_bbox

        # Data collect
        if success:
            pos_examples = self.pos_generator(
                self.target_bbox,
                self.opts["n_pos_update"],
                self.opts["overlap_pos_update"],
            )
            pos_feats = forward_samples(self.model, image, pos_examples, self.opts)
            self.pos_feats_all.append(pos_feats)
            if len(self.pos_feats_all) > self.opts["n_frames_long"]:
                del self.pos_feats_all[0]

            neg_examples = self.neg_generator(
                self.target_bbox,
                self.opts["n_neg_update"],
                self.opts["overlap_neg_update"],
            )
            neg_feats = forward_samples(self.model, image, neg_examples, self.opts)
            self.neg_feats_all.append(neg_feats)
            if len(self.neg_feats_all) > self.opts["n_frames_short"]:
                del self.neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(self.opts["n_frames_short"], len(self.pos_feats_all))
            pos_data = torch.cat(self.pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(self.neg_feats_all, 0)
            train(
                self.model,
                None,
                self.criterion,
                self.update_optimizer,
                pos_data,
                neg_data,
                self.opts["maxiter_update"],
                self.opts,
            )

        # Long term update
        elif self.current_frame % self.opts["long_interval"] == 0:
            pos_data = torch.cat(self.pos_feats_all, 0)
            neg_data = torch.cat(self.neg_feats_all, 0)
            train(
                self.model,
                self.model_g,
                self.criterion,
                self.update_optimizer,
                pos_data,
                neg_data,
                self.opts["maxiter_update"],
                self.opts,
            )

        torch.cuda.empty_cache()
        return bbreg_bbox
