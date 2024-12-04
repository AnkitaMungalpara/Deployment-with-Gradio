import lightning as L
import timm
import torch
from torchmetrics import Accuracy, MaxMetric, MeanMetric

class TimmClassifier(L.LightningModule):
    def __init__(
        self,
        base_model: str = "resnet18",
        num_classes: int = 4,
        pretrained: bool = True,
        lr: float = 1e-3,
        optimizer: str = "Adam",
        weight_decay: float = 1e-5,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # load pre-trained model
        self.model = timm.create_model(base_model, pretrained=pretrained, num_classes=num_classes)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch=batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch=batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)

        self.log(
            "val/acc_best",
            self.val_acc_best.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch=batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.hparams.optimizer)
        optimizer = optimizer_class(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
            eps=self.hparams.eps
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }
