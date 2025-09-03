import torch
from SITS.inference.base_infer import BaseInfer
from SITS.utils import INFER
import os
import yaml
import numpy as np
from dataset_utils.io_func import read_ENVI,write_ENVI,make_parent_dir
from dataset_utils.img_tool import img_norm_2
from SITS.utils import get_device
from einops import rearrange,reduce
from copy import deepcopy
import math
from osgeo import gdal
from osgeo import gdalconst
from tqdm import tqdm
import cv2
import time
@INFER.register_module()
class Singlevel_inference(BaseInfer):
    def __init__(self, model,meta):
        super(Singlevel_inference, self).__init__(model=model,meta=meta)

        self.image_path_root=self.meta["image_path_root"]
        self.parcel_path_root=self.meta["parcel_path_root"]
        self.yaml_path_root=self.meta["yaml_path_root"]
        self.tile_name=self.meta["tile_name"]
        self.num_classes=self.meta["num_classlevel"]
        self.patch_height,self.patch_width = self.meta["patch_size"]
        self.overlap= self.meta["over_lap_ratio"]
        self.doy = self.meta["doy"]
        self.save_path=self.meta["save_path"]
        self.model_mode=self.meta["model_mode"]
        self.color_map_dict=self.meta["color_map_dict"]
        self.label_level = self.meta["label_level"]

        self._load_checkpoint(self.meta["optimal_epoch"])
        self._create_save_single_dir()
        self._get_mean_std()
        self.model.to(self.device)

    def _build_device(self):
        print("sub class:")
        if torch.cuda.is_available():
            self.device_ids = [int(i) for i in self.meta["gpu_ids"]]
            self.device = get_device(self.device_ids, allow_cpu=False)
            self.len_device_ids = len(self.device_ids)
        else:
            self.device = torch.device('cpu')
        print(self.len_device_ids)
        print(self.device)
    def _get_mean_std(self):
        self.tile_mean_dict={}
        self.tile_std_dict={}
        for tile in self.tile_name:
            path_mean=os.path.join(self.yaml_path_root,  tile + "_mean.npz")
            path_std=os.path.join(self.yaml_path_root,  tile + "_std.npz")
            if os.path.exists(path_mean):
                self.tile_mean_dict[tile] = np.load(path_mean)["mean"]
            if os.path.exists(path_std):
                self.tile_std_dict[tile] = np.load(path_std)["std"]   
                

    def _create_save_single_dir(self):
        self.pred_path=os.path.join(self.save_path,"inference")
        if self.meta["parcel_constraint"]:
            type=["pixel","parcel"]
        else:
            type=["pixel"]
        for t in type:
            make_parent_dir(os.path.join(self.pred_path, "pred_"+t+"_level_" + f"{self.label_level}"+"/"))
            make_parent_dir(os.path.join(self.pred_path, "map_"+t+"_level_" + f"{self.label_level}"+"/"))
            make_parent_dir(os.path.join(self.pred_path, "mosaic_"+t+"_level_" + f"{self.label_level}"+"/"))
        make_parent_dir(os.path.join(self.pred_path, "prob_level_" + f"{self.label_level}"+"/"))
        make_parent_dir(os.path.join(self.pred_path, "truefalse_level_" + f"{self.label_level}" + "/"))

    def model_pred_singlevel_patch(self,subpatch,tile_name,return_fea=False):
        doy=np.asarray(self.doy[tile_name]).reshape(1, -1)
        subpatch = rearrange(subpatch, "h w (t c)-> 1 t c h w ", c=self.meta["num_feature"])
        subpatch = torch.from_numpy(subpatch).float().to(self.device)
        doy = torch.from_numpy(doy).to(self.device)
        self.model.eval()
        with torch.no_grad():
            # 是从单个类别回归到其他类别还是直接得到所有类别的预测结果
            if not return_fea:
                if self.cfg_model["type"] == "tsvit" or self.cfg_model["type"] == "utae" or self.cfg_model["type"] == "STNet"or self.cfg_model["type"] == "DBL" or self.cfg_model["type"] == "ms2ta":
                    prob = self.model.forward(subpatch, doy)
                elif self.cfg_model["type"] == "convrnn" or self.cfg_model["type"] == "BUnetConvLSTM":
                    prob = self.model.forward(subpatch, self.device)
                elif self.cfg_model["type"] == "unet3d" or self.cfg_model["type"] == "TFBS":
                    prob = self.model.forward(subpatch)
            else:
                if self.cfg_model["type"] == "tsvit" or self.cfg_model["type"] == "utae" or self.cfg_model["type"] == "ms2ta" or self.cfg_model["type"] == "dstan":
                    prob, feature = self.model.forward(subpatch, doy, return_fea=True)
                elif self.cfg_model["type"] == "convrnn" or self.cfg_model["type"] == "BUnetConvLSTM":
                    prob, feature = self.model.forward(subpatch, self.device, return_fea=True)
                elif self.cfg_model["type"] == "unet3d" or self.cfg_model["type"] == "TFBS":
                    prob, feature = self.model.forward(subpatch, return_fea=True)

            pred= prob.argmax(1)
            pred = pred.cpu().detach().numpy().squeeze()

            prob = prob.cpu().detach().numpy()
            prob = rearrange(prob, "1 c h w -> h w c")
            if return_fea:
                feature= feature.cpu().detach().numpy()

            if return_fea:
                if feature.ndim>4:
                    feature = reduce(feature, "1 c t h w -> c h w", "mean")
                    feature = rearrange(feature, "c h w -> h w c")
                else:
                    feature = rearrange(feature, "1 c h w -> h w c")
                return pred, prob, feature
            else:
                return pred,prob
            


    def _small_patch_inference(self,img_norm,tile_name):
        X_height,X_width=img_norm.shape[:2]

        pred_final=np.ones((X_height,X_width))*255
        prob_final=np.zeros((X_height,X_width,self.num_classes[0]))

        # the dimension change with the number of band
        
        if X_height <  self.patch_height or X_width <  self.patch_width:
            pad_height, pad_width = 0, 0
            if X_height < self.patch_height:
                pad_height = self.patch_height - X_height
            if X_width < self.patch_width:
                pad_width = self.patch_width - X_width
            img_norm = np.pad(img_norm, ((0, pad_height), (0, pad_width), (0, 0)))
            pred_final= np.pad(pred_final, ((0, pad_height), (0, pad_width)),constant_values=255)
            prob_final=np.pad(prob_final, ((0, pad_height), (0, pad_width), (0, 0)),constant_values=0)

        stride_width = math.ceil((1 - self.overlap) * self.patch_width)
        stride_height = math.ceil((1 - self.overlap) * self.patch_height)

        num_height = math.ceil((X_height - self.patch_height) / stride_height + 1)
        num_width = math.ceil((X_width - self.patch_width) / stride_width + 1)
        
        for i in range(num_height):
            if i < num_height - 1 or i==0:
                start_height = int(i * stride_height)
            else:
                start_height = int(X_height - self.patch_height)
            for j in range(num_width):
                if j < num_width - 1:
                    start_width = int(j * stride_width)
                else:
                    start_width = int(X_width - self.patch_width)
                subimage = img_norm[start_height:start_height + int(self.patch_height),
                           start_width:start_width + int(self.patch_width), :]

                # pred
                # 预测的小patch的结果
                pred_patch, prob_patch=self.model_pred_singlevel_patch(subimage,tile_name)

                    
                subpred = pred_final[start_height:start_height + int(self.patch_height),
                               start_width:start_width + int(self.patch_width)]

                subprob = prob_final[start_height:start_height + int(self.patch_height),
                                  start_width:start_width + int(self.patch_width),:]

                pred_fill = np.ones_like(subpred) * 255
                prob_fill = np.zeros_like(subprob)

                mask_1=subpred==255
                if np.any(mask_1):
                    pred_fill[mask_1]=pred_patch[mask_1]
                    prob_fill[mask_1]=prob_patch[mask_1]

                mask_2=subpred!=255
                if np.any(mask_2):
                    aver_prob=(subprob+prob_patch)/2
                    aver_pred=np.argmax(aver_prob,axis=-1)
                    pred_fill[mask_2]=aver_pred[mask_2]
                    prob_fill[mask_2] = aver_prob[mask_2]

                pred_final[start_height:start_height + int(self.patch_height),
                           start_width:start_width + int(self.patch_width)]=pred_fill

                prob_final[start_height:start_height + int(self.patch_height),
                                start_width:start_width + int(self.patch_width),:]=prob_fill
                                
        pred_final=pred_final[0:X_height,0:X_width]
        prob_final=prob_final[0:X_height,0:X_width]

        return pred_final,prob_final


    def _whole_patch_inference(self, img_norm, tile_name):
        X_height, X_width = img_norm.shape[:2]
        pred_patch, prob_patch= self.model_pred_singlevel_patch(img_norm, tile_name)
        return pred_patch, prob_patch


    def _parcel_constraint(self,pred_image,parcel_label):
        for i in np.unique(parcel_label).tolist():
            if i !=255:
                try:
                    field_indexes = parcel_label == i
                    pred = pred_image[field_indexes].astype(int)
                    pred = np.bincount(pred)
                    pred = np.argmax(pred)
                    pred_image[field_indexes] = pred
                except:
                    print("Index error")
                    print(pred_image.shape)
                    print(parcel_label.shape)

        return pred_image

    def _load_checkpoint(self,epoch):
        if torch.cuda.is_available() and self.len_device_ids > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)

            checkpoint = torch.load(os.path.join(self.save_path,  self.meta["model_name"]+'_epoch_'+str(epoch)+'_model.pth'))
            self.model.load_state_dict(checkpoint['network_state_dict'])

        if torch.cuda.is_available() and self.len_device_ids ==1:
            checkpoint = torch.load(os.path.join(self.save_path, self.meta["model_name"] + '_epoch_' + str(epoch) + '_model.pth'))['network_state_dict']
            new_checkpoint = {}
            for k,v in checkpoint.items():
                if "module" in k:
                    new_checkpoint[k.replace("module.","")]=v
                else:
                     new_checkpoint[k]=v
            self.model.load_state_dict(new_checkpoint)

    def get_extent(self,single_image_path):
        ds = gdal.Open(single_image_path)
        gt = ds.GetGeoTransform()
        # [经度，经度方向的分辨率，0（指北为0）,纬度，纬度方向的分辨率，0（指北为0）]

        return (gt[0], gt[3], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize)
    def _mosaic(self, image_dir, save_dir,label_dir,img_dir,img_idx):
        image_list = os.listdir(image_dir)
        min_x, max_y, max_x, min_y = self.get_extent(os.path.join(image_dir, image_list[0]))
        for file in image_list:
            
            # print(file)
            try:
                minx, maxy, maxx, miny = self.get_extent(os.path.join(image_dir, file))
            except:
                print(file)
            min_x = min(min_x, minx)
            max_y = max(max_y, maxy)
            max_x = max(max_x, maxx)
            min_y = min(min_y, miny)

        in_ds = gdal.Open(os.path.join(image_dir, image_list[0]))
        gt = in_ds.GetGeoTransform()
        rows = math.ceil((max_y - min_y) / -gt[5])
        columns = math.ceil((max_x - min_x) / gt[1])

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(save_dir, columns, rows)
        out_ds.SetProjection(in_ds.GetProjection())  # 设置投影
        out_bands = out_ds.GetRasterBand(1)

        initial_value = 255
        initial_array = np.full((rows, columns), initial_value, dtype=np.uint8)
        out_bands.WriteArray(initial_array)

        gt = list(in_ds.GetGeoTransform())
        gt[0], gt[3] = min_x, max_y
        out_ds.SetGeoTransform(gt)  # 设置坐标

        for file in image_list:
            label,_,_=read_ENVI(os.path.join(label_dir, file))
            label=label[:,:,-1]
            img, _, _ = read_ENVI(os.path.join(img_dir, file))
            img = img[:, :, -1]
            try:
                in_ds = gdal.Open(os.path.join(image_dir, file))
                trans = gdal.Transformer(in_ds, out_ds, [])
                success, xyz = trans.TransformPoint(False, 0, 0)
                x, y, z = map(int, xyz)
                data = in_ds.GetRasterBand(img_idx).ReadAsArray()
                data_b = out_bands.ReadAsArray(x, y, in_ds.RasterXSize, in_ds.RasterYSize)
                # if data_b.shape==3:
                #     data_b=data_b.transpose((1,2,0))
                #     data_b=data_b[:,:,3]
                mask = (data_b == 255)&(label!=255)&(img!=0)  # 没有值的
                # mask_repeat=data_b!=0
                # data_b[mask_repeat]=int((data_b[mask_repeat]+data[mask_repeat])/2)
                data_b[mask] = data[mask]
                out_bands.WriteArray(data_b, x, y)
            except:
                print("write error")
        del in_ds, out_ds, out_bands

    def visualize_features(self):
        if self.cfg_model["params"]["use_channel_fg"]:
            feature_channel=self.cfg_model["params"]["channel_group"]*self.num_classes[0]
        else:
            if self.cfg_model["type"]=="utae" or self.cfg_model["type"]=="BUnetConvLSTM":
                feature_channel=self.cfg_model["params"]["decoder_widths"][0]
            elif self.cfg_model["type"]=="unet3d":
                feature_channel=self.cfg_model["params"]["feats"]*2
            elif self.cfg_model["type"]=="TFBS":
                feature_channel=self.cfg_model["params"]["encoder_widths"][0]
            elif self.cfg_model["type"]=="convrnn":
                feature_channel=self.cfg_model["params"]["hidden_dim"]
    

        class_all_features=np.zeros((self.num_classes[0],self.meta["num_class_vis"],feature_channel))
        mark_each_class=np.zeros(self.num_classes[0])

        image_path_list = os.listdir(self.meta["test_data_path"])

        #shuffle the image_path_list
        np.random.shuffle(image_path_list)

        for data_name in tqdm(image_path_list):
            tile_name=data_name.split("-")[0]
            data_file_path = os.path.join(self.meta["test_data_path"], data_name)
            subpatch = np.load(data_file_path)["data"]
            label = np.load(data_file_path)["label"]
            label_cls=label[:,:,self.label_level-1]
            subpatch = rearrange(subpatch, " t c h w -> h w (t c)")
            _,_,sub_feature=self.model_pred_singlevel_patch(subpatch, tile_name,return_fea=True)

            
            unique_label = np.unique(label_cls)
            for i in unique_label:
                if i != 255:
                    if mark_each_class[i]<self.meta["num_class_vis"]:
                        fea_length=len(np.argwhere(label_cls == i))                      
                        sub_feature_cls=sub_feature[label_cls == i]
                        num_sub_fea=sub_feature_cls.shape[0]

                        #sub_feature_cls=np.mean(sub_feature_cls, axis=0,keepdims=True)
                        if int(mark_each_class[i])<self.meta["num_class_vis"]:
                            if int(mark_each_class[i] + num_sub_fea)<self.meta["num_class_vis"]:
                                class_all_features[int(i), int(mark_each_class[i]):int(mark_each_class[i] + num_sub_fea),:] = sub_feature_cls
                                mark_each_class[int(i)] = mark_each_class[int(i)] + num_sub_fea
                            else:
                                class_all_features[int(i), int(mark_each_class[i]):,:] = sub_feature_cls[:int(self.meta["num_class_vis"]-mark_each_class[i]),:]
                                mark_each_class[int(i)]=self.meta["num_class_vis"]
                            
        if not os.path.exists(self.meta["save_fea_path"] + "/"):
            os.makedirs(self.meta["save_fea_path"] + "/")
            
        np.savez(os.path.join(self.meta["save_fea_path"],self.meta["model_name"]+"_feature_vis_increase.npz"),class_all_features=class_all_features,mark_each_class=mark_each_class)

    def visualize_features_patch(self):
        # get features of each patch
        feature_patch_save_path=os.path.join(os.path.join(self.meta["save_fea_path"],self.meta["model_name"]+"_patch"))
        if not os.path.exists(feature_patch_save_path):
            os.makedirs(feature_patch_save_path)
        image_path_list = os.listdir(self.meta["test_data_path"])

        # shuffle the image_path_list
        np.random.shuffle(image_path_list)
        for data_name in tqdm(image_path_list):
            tile_name = data_name.split("-")[0]
            data_file_path = os.path.join(self.meta["test_data_path"], data_name)
            subpatch = np.load(data_file_path)["data"]
            subpatch = rearrange(subpatch, " t c h w -> h w (t c)")
            _, _, sub_feature = self.model_pred_singlevel_patch(subpatch, tile_name, return_fea=True)
            np.savez(os.path.join(feature_patch_save_path,data_name.split(".")[0]+".npz"),features=sub_feature)
            del subpatch, sub_feature
            
    def pred2colormap(self,color_map_dict,pred_file_path,color_file_path,nodata_value=255):
        dataset = gdal.Open(pred_file_path, gdalconst.GA_ReadOnly)
        band = dataset.GetRasterBand(1)  # 获取第一个波段
        raster_array = band.ReadAsArray()
        color_table = gdal.ColorTable()
        band.SetNoDataValue(nodata_value)
        # 添加颜色到颜色映射表
        for key, map_value in color_map_dict.items():
            color_table.SetColorEntry(int(key), (map_value[0], map_value[1], map_value[2]))  # 设置索引为0的颜色

        # 将颜色映射表应用于栅格波段
        band.SetRasterColorTable(color_table)
        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.CreateCopy(color_file_path, dataset)
        output_dataset = None  # 释放资源
        dataset = None


    def infer_image(self):
        image_path_list = os.listdir(self.image_path_root)
        # num_class_new = [0]
        # num_class_new.extend(num_class_new[i] + self.num_classes[i] for i in range(len(self.num_classes)))
        
        all_time=0
        for image_name in tqdm(image_path_list):
            file_path_1 = os.path.join(self.pred_path, "pred_pixel_level_" + f"{self.label_level}")
            file_path_2 = os.path.join(self.pred_path, "pred_parcel_level_" + f"{self.label_level}")
            check_path = os.path.exists(os.path.join(file_path_1, image_name)) and os.path.exists(os.path.join(file_path_2, image_name))
#            if check_path:
#                continue
#            else:
            img_file_path=os.path.join(self.image_path_root,image_name)
            parcel_path = os.path.join(self.parcel_path_root, image_name)

            tile_name=image_name.split("-")[0]

            mean=self.tile_mean_dict[tile_name]
            std=self.tile_std_dict[tile_name]

            img, trans, proj = read_ENVI(img_file_path)
            img_norm = img_norm_2(img, mean,std)
            if self.meta["parcel_constraint"] and os.path.exists(parcel_path):
                parcel_label,_,_=read_ENVI(parcel_path)

            start_time=time.time()
            if self.meta["small_patch"]:
                pred_image, prob_image=self._small_patch_inference(img_norm,tile_name)
            else:
                pred_image, prob_image = self._whole_patch_inference(img_norm,tile_name)

            end_time=time.time()-start_time
            print(f"predict one image time: {end_time:.4f} seconds,{end_time/60:4f} minutes,{end_time/3600:4f} hours")
            all_time=all_time+end_time

            file_path = os.path.join(self.pred_path, "pred_pixel_level_" + f"{self.label_level}")
            write_ENVI(os.path.join(file_path, image_name), pred_image, trans, proj)
            file_path = os.path.join(self.pred_path, "prob_level_" + f"{self.label_level}")
            write_ENVI(os.path.join(file_path, image_name), prob_image, trans, proj)

            if self.meta["parcel_constraint"]:
                # print("image shape:")
                # print(img_norm.shape)
                # print(pred_image.shape)
                # print(parcel_label.shape)
                pred_image_parcel=self._parcel_constraint(pred_image,parcel_label[:,:,4])
                file_path = os.path.join(self.pred_path, "pred_parcel_level_" + f"{self.label_level}")
                write_ENVI(os.path.join(file_path, image_name), pred_image_parcel, trans, proj)

        print(f"predict all images time: {all_time:.4f} seconds,{all_time/60:4f} minutes,{all_time/3600:4f} hours")

    def mosaic_image(self):
        mosaic_image_path=os.path.join(self.pred_path, "pred_pixel_level_" + f"{self.label_level}")
        save_path=os.path.join(os.path.join(self.pred_path, "mosaic_pixel_level_" + f"{self.label_level}"),self.meta["area"]+"_"+self.meta["model_name"]+"_level_"+f"{self.label_level}"+".tif")
        self._mosaic(mosaic_image_path,save_path,self.meta["parcel_path_root"],self.meta["image_path_root"],1)
        if self.meta["parcel_constraint"]:
            mosaic_image_path = os.path.join(self.pred_path, "pred_parcel_level_" + f"{self.label_level}")
            save_path = os.path.join(os.path.join(self.pred_path, "mosaic_parcel_level_" + f"{self.label_level}"),
                                     self.meta["area"] + "_" + self.meta["model_name"] + "_level_" + f"{self.label_level}" + ".tif")
            self._mosaic(mosaic_image_path, save_path, self.meta["parcel_path_root"], self.meta["image_path_root"], 1)


    def pred2color_image(self):

        color_map_dict= self.color_map_dict[self.label_level]
        pred_file_path = os.path.join(os.path.join(self.pred_path, "mosaic_pixel_level_" + f"{self.label_level}"), self.meta["area"]+"_"+self.meta["model_name"] + "_level_" + f"{self.label_level}" + ".tif")
        color_file_path = os.path.join(os.path.join(self.pred_path, "map_pixel_level_" + f"{self.label_level}"), self.meta["area"]+"_"+self.meta["model_name"] + "_level_" + f"{self.label_level}" + "_color.tif")
        self.pred2colormap(color_map_dict, pred_file_path, color_file_path)
        if self.meta["parcel_constraint"]:
            pred_file_path = os.path.join(os.path.join(self.pred_path, "mosaic_parcel_level_" + f"{self.label_level}"), self.meta["area"]+"_"+self.meta["model_name"] + "_level_" + f"{self.label_level}" + ".tif")
            color_file_path = os.path.join(os.path.join(self.pred_path, "map_parcel_level_" + f"{self.label_level}"), self.meta["area"]+"_"+self.meta["model_name"] + "_level_" + f"{self.label_level}" + "_color.tif")
            self.pred2colormap(color_map_dict, pred_file_path, color_file_path)
            

    def truelabel2color_image(self):
        #get color map of the true label
        color_map_dict = self.color_map_dict[self.label_level]
        image_path_list = os.listdir(self.meta["test_data_path"])
        for data_name in tqdm(image_path_list):
            data_file_path = os.path.join(self.meta["test_data_path"], data_name)
            label = np.load(data_file_path)["label"][:,:,self.meta["label_level"]-1]
            rgb_label = np.array([color_map_dict[val] for row in label for val in row], dtype=np.uint8)
            rgb_label = rgb_label.reshape((label.shape[0], label.shape[1], 3))
            save_file_path = os.path.join(self.meta["test_label_rgb_path"], data_name.split(".")[0]+".jpg")
            cv2.imwrite(save_file_path, rgb_label)
            

    def True_false_map(self):
        pred_parcel_save_path = os.path.join(os.path.join(self.pred_path, "mosaic_parcel_level_" + f"{self.label_level}"),
                                 self.meta["area"] + "_" + self.meta[
                                     "model_name"] + "_level_" + f"{self.label_level}" + ".tif")
        label_parcel_save_path = os.path.join(os.path.join(self.meta["true_label_path"],"mosaic_" + f"{self.label_level}"), self.meta["area"]+"_label_level_3" + ".tif")

        label_truefalse_save_path = os.path.join(os.path.join(self.pred_path, "truefalse_level_" + f"{self.label_level}"),
                                              self.meta["area"] + "_" + self.meta["model_name"]+"_truefalse_level_3" + ".tif")
        print(pred_parcel_save_path)
        print(label_parcel_save_path)
        pred_parcel,trans,proj=read_ENVI(pred_parcel_save_path)
        label_parcel,_,_=read_ENVI(label_parcel_save_path)
        
        print(pred_parcel.shape)
        print(label_parcel.shape)
        pred_parcel=pred_parcel[:label_parcel.shape[0],:]
        print(pred_parcel.shape)
        print(label_parcel.shape)
        truefalse=np.ones_like(pred_parcel)*255
        mask=(pred_parcel==label_parcel)&(label_parcel!=255)
        truefalse[mask]=1
        mask = (pred_parcel != label_parcel) & (label_parcel != 255)
        truefalse[mask] = 2
        write_ENVI(label_truefalse_save_path,truefalse,trans,proj)
        self.pred2colormap(self.meta["true_false_color_dict"], label_truefalse_save_path, label_truefalse_save_path.replace(".tif","_color.tif"))

