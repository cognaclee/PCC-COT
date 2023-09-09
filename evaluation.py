import os
import sys
import torch
from glob import glob
import numpy as np
import open3d as o3d
from metrics.PSNR import get_psnr2
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
import plyfile

chamfer_dist = chamfer_3DDist()


def read_pcd_from_ply(fileName):
    f = open(fileName)
    for i in range(9):
        head = f.readline()
    points = []
    for line in f.readlines():
        line = line.strip('\n') 
        oneline = line.split(' ')
        points.append([float(oneline[0]),float(oneline[1]),float(oneline[2])])
    points = np.array(points, dtype=np.float32)
    return points
    
def read_pcd_from_ply2(fileName):
    loaded = plyfile.PlyData.read(fileName)
    points = np.vstack([loaded['vertex'].data['x'], loaded['vertex'].data['y'], loaded['vertex'].data['z']])
    return points
    
def cal_cd_psnr_bpp(gt_dir,other_dir,peak):  
  dir_nums = len(other_dir)
  gt_names = sorted(glob(gt_dir+'/*.'+'ply'))
  pc_nums = len(gt_names)
  cd_metric = np.zeros((dir_nums,pc_nums))
  psnr_metric = np.zeros((dir_nums,pc_nums))
  
  start = len(gt_dir)
  for i in range(pc_nums):
      gt_name = gt_names[i]
      print('gt_name= ',gt_name)
      gt = read_pcd_from_ply(gt_name)
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(gt)
      # estimate the normal
      pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=12))
      pcd.normalize_normals()
      gt_normals = np.array(pcd.normals)
      gt = torch.tensor(gt).float().cuda().unsqueeze(0)
      gt_normals = torch.tensor(gt_normals).float().cuda().unsqueeze(0)
      #print('gt.shape=',gt.shape)
      for j in range(dir_nums):
          name = other_dir[j] + gt_name[start:]
          points = read_pcd_from_ply(name)
          points = torch.tensor(points).float().cuda().unsqueeze(0)
          gt2pred_loss, pred2gt_loss, _, _ = chamfer_dist(gt, points)
          cd = gt2pred_loss.mean() + pred2gt_loss.mean()
          psnr = get_psnr2(gt, gt_normals, points, peak)
          cd_metric[j,i] = cd.item()
          psnr_metric[j,i] = psnr.item()
  return cd_metric, psnr_metric, gt_names


if __name__ == '__main__':
    gt_dir = './output/2022-10-14T13:42:04.667891/pcd/merge/gt/'
    other_dir = ['/data/PCC_Results/ours/']
    other_dir.append('/data/PCC_Results/D-PCC/')
    other_dir.append('/data/PCC_Results/mpeg/')
    other_dir.append('/data/PCC_Results/draco/')
    
    peak = 1.2518049478530884 # shapenet:0.7965023517608643   /semantickitti:1.2518049478530884
    cd_metric, psnr_metric, gt_names = cal_cd_psnr_bpp(gt_dir,other_dir,peak)
    
    fo = open("metrics.txt", "w")
    seq = ["fileBaseName", "       ours      ", "       D-PCC      ", "       mpeg      ", "       draco\n"]
    fo.writelines( seq )
    pc_nums = len(gt_names)
    cd_m = 100000
    ps_m = 100
    start = len(gt_dir)
    for i in range(pc_nums):
        cd0=round(cd_m*cd_metric[0,i])/cd_m
        ps0=round(ps_m*psnr_metric[0,i])/ps_m
        cd0_ps0 = '  '+str(cd0)+'/'+str(ps0)+'  '
        
        cd1=round(cd_m*cd_metric[1,i])/cd_m
        ps1=round(ps_m*psnr_metric[1,i])/ps_m
        cd1_ps1 = '  '+str(cd1)+'/'+str(ps1)+'  '
        
        cd2=round(cd_m*cd_metric[2,i])/cd_m
        ps2=round(ps_m*psnr_metric[2,i])/ps_m
        cd2_ps2 = '  '+str(cd2)+'/'+str(ps2)+'  '
        
        cd3=round(cd_m*cd_metric[3,i])/cd_m
        ps3=round(ps_m*psnr_metric[3,i])/ps_m
        cd3_ps3 = '  '+str(cd3)+'/'+str(ps3)+'\n'
        
        seq = [gt_names[i][start:],'      ', cd0_ps0, cd1_ps1, cd2_ps2, cd3_ps3]
        fo.writelines(seq)
    fo.close()
