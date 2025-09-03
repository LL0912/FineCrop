import os
import torch
from tqdm import tqdm
import numpy as np
import wandb
from SITS.apis.base_trainer import BaseTrainer
from SITS.datasets.imagedataset import build_unequal_dataloader
from SITS.models import build_model
from SITS.loss_functions import build_loss_functions
from SITS.optimization import build_optimizer, build_lr_scheduler
from SITS.utils import TRAINER
from SITS.utils import get_device
from SITS.utils import build_confusion_matrix, confusion_matrix_to_accuraccies
import time
from thop import profile
"""
parcel region group and
channel devil
"""
@TRAINER.register_module(name="UnequalImage_SinglevelTrainer_womap_pr_cd")
class UnequalImage_SinglevelTrainer_womap_pr_cd(BaseTrainer):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(UnequalImage_SinglevelTrainer_womap_pr_cd, self).__init__(trainer=trainer, dataset=dataset, model=model,
                                                                  loss_function=loss_function,
                                                                  optimizer=optimizer, lr_scheduler=lr_scheduler,
                                                                  meta=meta)
        self.build_scalar_recoder()

    def _build_train_dataloader(self):
        return build_unequal_dataloader(self.cfg_dataset['train'], self.cfg_dataset["batch_size"],
                                        self.cfg_dataset["num_workers"])
        # return build_unequal_dataloader(self.cfg_dataset['train'], self.cfg_dataset["batch_size"], self.cfg_dataset["num_workers"], shuffle=True)

    def _build_val_dataloader(self):
        return build_unequal_dataloader(self.cfg_dataset['val'], self.cfg_dataset["batch_size"],
                                        self.cfg_dataset["num_workers"])

    def _build_test_dataloader(self):
        return build_unequal_dataloader(self.cfg_dataset['test'], self.cfg_dataset["batch_size"],
                                        self.cfg_dataset["num_workers"])

    def _build_model(self):
        return build_model(self.cfg_model)

    def _build_loss_function(self):
        return build_loss_functions(**self.cfg_loss_function)

    def _build_optimizer(self):
        self.cfg_optimizer['params'].update(params=self.model.parameters())
        return build_optimizer(self.cfg_optimizer)

    def _build_lr_scheduler(self):
        self.cfg_lr_scheduler['params'].update(optimizer=self.optimizer)
        return build_lr_scheduler(self.cfg_lr_scheduler)

    def _build_device(self):
        if torch.cuda.is_available():
            self.device_ids = [int(i) for i in self.cfg_trainer["params"]["gpu_ids"]]
            self.device = get_device(self.device_ids, allow_cpu=False)
            self.len_device_ids = len(self.device_ids)
        else:
            self.device = torch.device('cpu')

    def build_scalar_recoder(self):
        if self.meta["log_way"] == "scaler":
            self.loss_recorder = self._build_scalar_recoder()
            self.grad_l2_norm = self._build_scalar_recoder()

        if self.meta["log_way"] == "wandb":
            wandb.init(
                # set the wandb project where this run will be logged
                project="FGCrop",
                entity="aiagri",
                name=self.meta["model_name"],
                # track hyperparameters and run metadata
                config={
                    "learning_rate": self.cfg_optimizer["params"]["lr"],
                    "architecture": self.cfg_model["type"],
                    "dataset": self.meta["area"],
                    "epochs": self.cfg_trainer["params"]["max_iters"],
                    "num_class": self.cfg_model["params"]["num_classlevel"]
                }
            )

        self.val_acc_pixel = self._build_dict_recorder()
        self.val_acc_field = self._build_dict_recorder()

        self.test_acc_pixel = self._build_dict_recorder()
        self.test_acc_field = self._build_dict_recorder()

    def fine_grained_2_multilevel(self, map_dict, z4):
        map_dict_3 = map_dict[3]
        map_dict_2 = map_dict[2]
        map_dict_1 = map_dict[1]
        z1 = torch.ones_like(z4) * 255
        z2 = torch.ones_like(z4) * 255
        z3 = torch.ones_like(z4) * 255
        for i_parent, id_list in map_dict_1.items():
            for id in id_list:
                mask = z4 == id
                z1[mask] = i_parent
        for i_parent, id_list in map_dict_2.items():
            for id in id_list:
                mask = z4 == id
                z2[mask] = i_parent
        for i_parent, id_list in map_dict_3.items():
            for id in id_list:
                mask = z4 == id
                z3[mask] = i_parent
        return z1, z2, z3

    def _validation(self, epoch, dataloader, val_mode="val"):
        # for i in range(1,self.cfg_model["params"]["num_labelevel"]+1):
        #     locals()["cm_pixel_"+f"{i}"+"_all"] = np.zeros((self.cfg_model["params"]["num_classlevel"][i-1], self.cfg_model["params"]["num_classlevel"][i-1]))
        #     locals()["cm_field_"+f"{i}"+"_all"] = np.zeros((self.cfg_model["params"]["num_classlevel"][i-1], self.cfg_model["params"]["num_classlevel"][i-1]))

        cm_pixel_all = np.zeros((self.meta["num_classlevel"][self.meta["train_level"] - 1],
                                 self.meta["num_classlevel"][self.meta["train_level"] - 1]))

        cm_field_all = np.zeros(
            (self.meta["num_classlevel"][self.meta["train_level"] - 1],
             self.meta["num_classlevel"][self.meta["train_level"] - 1]))

        self.model.eval()
        with torch.no_grad():
            if self.cfg_dataset["train"]["params"]["num_level"] == 4:
                for iteration, data in tqdm(enumerate(dataloader)):
                    inputs, target_level_1, target_level_2, target_level_3, target_level_4, gt_instance, doy, _ = data
                    del data
                 
                    if torch.cuda.is_available():
                        inputs = inputs.to(self.device)
                        doy = doy.to(self.device)

                    if self.cfg_model["type"] == "tsvit" or self.cfg_model["type"] == "utae"  or self.cfg_model["type"] == "DBL"or self.cfg_model["type"] == "STNet"or self.cfg_model["type"] == "ms2ta":
                        outputs = self.model.forward(inputs, doy)

                    elif self.cfg_model["type"] == "convrnn" or self.cfg_model["type"] == "BUnetConvLSTM":
                        outputs = self.model.forward(inputs, self.device)
                    elif self.cfg_model["type"] == "unet3d" or self.cfg_model["type"] == "TFBS":
                        outputs = self.model.forward(inputs)

                    if self.meta["train_level"] == 4:
                        target_label = target_level_4
                    elif self.meta["train_level"] == 3:
                        target_label = target_level_3
                    elif self.meta["train_level"] == 2:
                        target_label = target_level_2
                    elif self.meta["train_level"] == 1:
                        target_label = target_level_1

                    outputs = outputs.argmax(1)
                    outputs = outputs.cpu().detach().numpy()
                  
                    cm_pixel = self.evaluation_pixel(outputs, target_label.numpy(), self.meta["ignore_index"],
                                                     self.meta["num_classlevel"][self.meta["train_level"] - 1])

                    cm_pixel_all += cm_pixel
                    cm_field = self.evaluation_field(outputs, target_label.numpy(), self.meta["ignore_index"],
                                                     gt_instance,
                                                     self.meta["num_classlevel"][self.meta["train_level"] - 1])

                    cm_field_all += cm_field
        # 对总体的矩阵计算各类精度评价指标，返回评价指标，以及总体的混淆矩阵

        acc_pixel = confusion_matrix_to_accuraccies(cm_pixel_all)
        acc_field = confusion_matrix_to_accuraccies(cm_field_all)

        if val_mode == "val":
            self.val_acc_pixel.update_dict(acc_pixel)
            self.val_acc_field.update_dict(acc_field)

        if val_mode == "test":
            self.test_acc_pixel.update_dict(acc_pixel)
            self.test_acc_field.update_dict(acc_field)

    def _train(self, epoch):
        
        self.model.train()
        # 需要记录的训练的loss
        mean_loss_total = 0
        # 开始训练
        start_time = time.time()
        for iteration, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()


            input, target_local_1, target_local_2, target_local_3, target_local_4, instance, doy, _ = data
            if torch.cuda.is_available():
                input = input.to(self.device)
                doy = doy.to(self.device)
            
            if self.meta["train_level"] == 4:
                target_label = target_local_4.to(self.device)
            elif self.meta["train_level"] == 3:
                target_label = target_local_3.to(self.device)
            elif self.meta["train_level"] == 2:
                target_label = target_local_2.to(self.device)
            elif self.meta["train_level"] == 1:
                target_label = target_local_1.to(self.device)


            if self.cfg_model["type"] == "tsvit" or self.cfg_model["type"] == "utae" or self.cfg_model["type"] == "DBL" or self.cfg_model["type"] == "STNet"or self.cfg_model[
                "type"] == "ms2ta" or self.cfg_model["type"] == "dstan":
                if self.meta["train_mode"].split("_")[2] == "cd" or self.meta["train_mode"].split("_")[2] == "cdce":
                    output_local_pred,feature = self.model.forward(input, doy,return_fea=True)
                    
                else:
                    output_local_pred = self.model.forward(input, doy)

            elif self.cfg_model["type"] == "convrnn" or self.cfg_model["type"] == "BUnetConvLSTM":
                if self.meta["train_mode"].split("_")[2] == "cd" or self.meta["train_mode"].split("_")[2] == "cdce":
                    output_local_pred,feature = self.model.forward(input, self.device,return_fea=True)
                else:
                    output_local_pred = self.model.forward(input, self.device)    
                    
            elif self.cfg_model["type"] == "unet3d" or self.cfg_model["type"] == "TFBS":
                if self.meta["train_mode"].split("_")[2] == "cd" or self.meta["train_mode"].split("_")[2] == "cdce":
                    output_local_pred,feature = self.model.forward(input, return_fea=True)     
                else:
                    output_local_pred = self.model.forward(input)


            #分配类别的数量信息
            if "pr" in self.meta["train_mode"].split("_")[0]:
                cls_num_list = self.meta["parcel_each_level"]["l" + str(self.meta["train_level"])]
            else:
                cls_num_list = self.meta["num_each_level"]["l" + str(self.meta["train_level"])]


            if self.meta["train_mode"].split("_")[1] == "dw":
                idx = epoch // (self.cfg_trainer["" "params"]["max_iters"] // 2 + 1)
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                per_cls_weights = torch.from_numpy(per_cls_weights).double().to(self.device)
            elif self.meta["train_mode"].split("_")[1] != "dw":
                per_cls_weights = None


            # 计算loss
            if self.meta["train_mode"].split("_")[0] == "lt":
                total_loss = self.loss_lt(output_local_pred, target_label, np.asarray(cls_num_list), self.device,
                                          weight=per_cls_weights)
            elif self.meta["train_mode"].split("_")[0] == "ltpr":
                instance=instance.to(self.device)
                total_loss = self.cfg_trainer["params"]["lambda_lt"] * self.loss_lt(output_local_pred, target_label,
                                                                                    instance,
                                                                                    np.asarray(cls_num_list),
                                                                                    self.device,
                                                                                    weight=per_cls_weights)
            elif self.meta["train_mode"].split("_")[0] == "ce":
                total_loss = self.loss_ce(output_local_pred, target_label)

            if self.meta["train_mode"].split("_")[2] == "cdce":
                #instance=instance.to(self.device)
                l_dis,l_div=self.loss_cd(feature, target_label,self.device)
                total_loss += self.cfg_trainer["params"]["lambda_cd_alpha"]*l_dis+self.cfg_trainer["params"]["lambda_cd_beta"]*l_div
                total_loss += self.loss_ce(output_local_pred, target_label)
            elif self.meta["train_mode"].split("_")[2] == "cd":
                instance=instance.to(self.device)
                l_dis, l_div = self.loss_cd(feature, target_label, instance,self.device)
                total_loss += self.cfg_trainer["params"]["lambda_cd_alpha"] * l_dis + self.cfg_trainer["params"]["lambda_cd_beta"] * l_div
            elif self.meta["train_mode"].split("_")[2] == "ce":
                total_loss += self.loss_ce(output_local_pred, target_label)


            total_loss.backward()
            mean_loss_total += total_loss.detach().cpu().numpy()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg_trainer["params"]["grad_clip"])
            # self.clip_recoder_grad_norm(model=self.model,grad_l2_norm=self.grad_l2_norm)
            self.optimizer.step()

            if iteration%2000==0:
                print(mean_loss_total / (iteration+1))

        self.lr_scheduler.step(epoch=epoch)

        elapsed_time = time.time() - start_time

        print(f"one epoch time: {elapsed_time:.4f} seconds,{elapsed_time/60:4f} minutes,{elapsed_time/3600:4f} hours ")

        print("total loss: ", mean_loss_total / (iteration+1))
        if self.meta["log_way"] == "scaler":
            self.loss_recorder.update_scalar(mean_loss_total / (iteration + 1))

        elif self.meta["log_way"] == "wandb":
            wandb.log({"Training total loss": mean_loss_total / (iteration + 1)})

    def _save_checkpoint(self, epoch):
        torch.save({'network_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   os.path.join(self.save_path, self.meta["model_name"] + '_epoch_' + str(epoch) + '_model.pth'))

    def _load_checkpoint(self, epoch, new_model=False):
        if torch.cuda.is_available() and self.len_device_ids > 1:
            checkpoint = torch.load(
                os.path.join(self.save_path, self.meta["model_name"] + '_epoch_' + str(epoch) + '_model.pth'))
            self.model.load_state_dict(checkpoint['network_state_dict'])

        if torch.cuda.is_available() and self.len_device_ids == 1:
            checkpoint = torch.load(
                os.path.join(self.save_path, self.meta["model_name"] + '_epoch_' + str(epoch) + '_model.pth'))
            model_checkpoint = checkpoint['network_state_dict']
            if new_model:
                new_checkpoint = {}
                for k, v in model_checkpoint.items():
                    if "module" in k:
                        new_checkpoint[k.replace("module.", "")] = v
                self.model.load_state_dict(new_checkpoint)
            else:
                self.model.load_state_dict(model_checkpoint)

        checkpoint = torch.load(
            os.path.join(self.save_path, self.meta["model_name"] + '_epoch_' + str(epoch) + '_model.pth'))
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def _build_components(self):
        self.train_loader = self._build_train_dataloader()
        self.val_loader = self._build_val_dataloader()
        self.test_loader = self._build_test_dataloader()
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        if self.meta["train_mode"]== "ce_no_no":
            self.loss_ce = self._build_loss_function()[0]
        else:
            self.loss_ce, self.loss_lt,self.loss_cd = self._build_loss_function()
        

        if torch.cuda.is_available() and self.len_device_ids > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)


        if self.meta["best_iter"] != self.meta["max_iter"]:
            self._load_checkpoint(self.meta["best_iter"])
            
        self.model.to(self.device)

        print(f'model name: {self.meta["model_name"]}')
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params_m = total_params / 1e6  # 转换为兆
        print(f'model parameter: {total_params_m:.2f} MB')

        total_memory = total_params * 4 / (1024 ** 2)  # 以MB为单位，假设每个参数为4字节（float32）
        print(f'model gpus: {total_memory:.2f} MB')
        
        
        
    def evaluation_pixel(self, log_pro, targets, ignore_index, num_class):
        # log_pro = log_pro.argmax(1)
        predictions = log_pro.flatten()
        targets = targets.flatten()
        valid_mask = targets != ignore_index

        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        confusion_matrix_pixel = build_confusion_matrix(targets, predictions, num_class, ignore_index)
        return confusion_matrix_pixel

    def evaluation_field(self, log_pro, targets, ignore_index, gt_instance, num_class):
        # log_pro = log_pro.argmax(1)
        predictions = log_pro.flatten()
        targets = targets.flatten()
        gt_instance = gt_instance.flatten()

        valid_mask = targets != ignore_index
        targets_mask = targets[valid_mask]
        gt_instance_mask = gt_instance[valid_mask]
        predictions_mask = predictions[valid_mask]

        prediction_fieldwise = np.zeros_like(targets_mask)

        for i in np.unique(gt_instance_mask).tolist():
            field_indexes = gt_instance_mask == i
            try:
                pred = predictions_mask[field_indexes]
                pred = np.bincount(pred)
                pred = np.argmax(pred)
                prediction_fieldwise[field_indexes] = pred
            except IndexError:
                print(predictions_mask.shape)
                print(field_indexes.shape)
        
        # with open('/data1/ll20/Pub_dataset/EuroCrops/NL_select/result/l3/utae_fg/pixel_result.txt', 'a') as f:
        #     np.savetxt(f, predictions_mask.reshape(-1, 1), fmt='%d', delimiter=',')
        # with open('/data1/ll20/Pub_dataset/EuroCrops/NL_select/result/l3/utae_fg/parcel_result.txt', 'a') as f:
        #     np.savetxt(f, prediction_fieldwise.reshape(-1, 1), fmt='%d', delimiter=',')
        # with open('/data1/ll20/Pub_dataset/EuroCrops/NL_select/result/l3/utae_fg/target.txt', 'a') as f:
        #     np.savetxt(f, targets_mask.reshape(-1, 1), fmt='%d', delimiter=',')
        # with open('/data1/ll20/Pub_dataset/EuroCrops/NL_select/result/l3/utae_fg/parcel_id.txt', 'a') as f:
        #     np.savetxt(f, gt_instance_mask.reshape(-1, 1), fmt='%d', delimiter=',')

        confusion_matrix_pixel = build_confusion_matrix(targets_mask, prediction_fieldwise, num_class, ignore_index)

        return confusion_matrix_pixel
    def _get_accuracy(self,mode,scale):
        acc_obj=getattr(self,f"{mode}_acc_{scale}")
        records = acc_obj.get_records()
        acc, kappa, mf1, miou = (records[key][-1] for key in ("acc", "kappa", "mf1", "miou"))
        return acc, kappa, mf1, miou
    def run(self, validation=True):

        self.best_pixel_acc = -1
        self.best_pixel_miou = -1
        self.best_epoch = 0

        if self.meta["best_iter"] ==  self.meta["max_iter"]:
            bar = tqdm(list(range(1, self.cfg_trainer['params']['max_iters'] + 1)))
        else:
            bar = tqdm(list(range(self.meta["best_iter"] + 1, self.cfg_trainer['params']['max_iters'] + 1)))

        for i in bar:
            self._train(i)
            if validation:
                self._validation(self.meta["best_iter"] + i, self.val_loader, "val")
                acc_mean = self.val_acc_pixel.get_records()["acc"][-1]
                acc_miou = self.val_acc_pixel.get_records()["miou"][-1]
                if i > 1 and acc_mean > self.best_pixel_acc and acc_miou > self.best_pixel_miou:
                    self.best_pixel_acc = acc_mean
                    self.best_pixel_miou = acc_miou
                    self.best_epoch = i
                    self._save_checkpoint(i)
                print("model name:",self.meta["model_name"])
                acc, kappa, mf1, miou = self._get_accuracy("val","pixel")
                print(f"val acc on pixel:{acc:.4f},kappa:{kappa:.4f},mf1:{mf1:.4f},miou:{miou:.4f}")
                acc, kappa, mf1, miou = self._get_accuracy("val","field")
                print(f"val acc on field:{acc:.4f},kappa:{kappa:.4f},mf1:{mf1:.4f},miou:{miou:.4f}")

        self._save_checkpoint(self.cfg_trainer['params']['max_iters'])

        final_epoch = self.best_epoch if self.best_epoch != 0 else self.cfg_trainer['params']['max_iters']
        self._load_checkpoint(final_epoch, new_model=False)
        self._validation(self.cfg_trainer['params']['max_iters'], self.test_loader, "test")
        acc = self.test_acc_pixel.get_records()["acc"][-1]
        kappa = self.test_acc_pixel.get_records()["kappa"][-1]
        mf1 = self.test_acc_pixel.get_records()["mf1"][-1]
        miou = self.test_acc_pixel.get_records()["miou"][-1]
        print(f"test acc:{acc:.4f},kappa:{kappa:.4f},mf1:{mf1:.4f},miou:{miou:.4f}")
        self.test_acc_pixel.save_dict_to_excel(os.path.join(self.save_path,
                                                            self.meta["model_name"] + "_test_acc_pixel_" + str(
                                                                self.meta["train_level"]) + ".xlsx"))
        self.test_acc_field.save_dict_to_excel(os.path.join(self.save_path,
                                                            self.meta["model_name"] + "_test_acc_field_" + str(
                                                                self.meta["train_level"]) + ".xlsx"))
        self.val_acc_pixel.save_dict_to_excel(os.path.join(self.save_path,
                                                            self.meta["model_name"] + "_val_acc_pixel_" + str(
                                                                self.meta["train_level"]) + ".xlsx"))
        self.val_acc_field.save_dict_to_excel(os.path.join(self.save_path,
                                                            self.meta["model_name"] + "_val_acc_field_" + str(
                                                                self.meta["train_level"]) + ".xlsx"))
        if self.meta["log_way"] == "scaler":
            self.loss_recorder.save_scalar_npy("train_loss", self.save_path)
        


    def test(self):
        # load the model
        # self._load_checkpoint(self.meta["best_iter"], True)
        self._validation(self.meta["best_iter"], self.test_loader, "test")
        self.test_acc_pixel.save_dict_to_excel(os.path.join(self.save_path,
                                                            self.meta["model_name"] + "_test_acc_pixel_" + str(
                                                                self.meta["train_level"]) +"_epoch_"+str(self.meta["best_iter"])+ ".xlsx"))
        self.test_acc_field.save_dict_to_excel(os.path.join(self.save_path,
                                                            self.meta["model_name"] + "_test_acc_field_" + str(
                                                                self.meta["train_level"]) +"_epoch_"+str(self.meta["best_iter"])+ ".xlsx"))
        acc, kappa, mf1, miou = self._get_accuracy("test","pixel")
        print(f"val acc on pixel:{acc:.4f},kappa:{kappa:.4f},mf1:{mf1:.4f},miou:{miou:.4f}")
        acc, kappa, mf1, miou = self._get_accuracy("test","field")
        print(f"val acc on field:{acc:.4f},kappa:{kappa:.4f},mf1:{mf1:.4f},miou:{miou:.4f}")
    
    

