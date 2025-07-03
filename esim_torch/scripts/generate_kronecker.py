import argparse
from operator import sub
import os
import esim_torch
import numpy as np
import glob
import cv2
import tqdm
import torch
from os import path
from os import listdir
from os.path import  join, isfile
from pprint import pprint

def ensure_dir(target_path):
    if not path.exists(target_path):
        os.makedirs(target_path)
def is_valid_dir(subdirs, files):
    return len(subdirs) == 0 and len(files) >100


def process_dir(outdir, indir, args):
    print(f"Processing folder {indir}... Generating events in {outdir}")
    os.makedirs(outdir, exist_ok=True)

    # constructor
    esim = esim_torch.ESIM(args.contrast_threshold_negative,
                           args.contrast_threshold_positive,
                           args.refractory_period_ns)

    timestamps = np.genfromtxt(os.path.join(indir, "timestamps.txt"), dtype="float64")
    timestamps_ns = (timestamps * 1e9).astype("int64")
    timestamps_ns = torch.from_numpy(timestamps_ns).cuda()

    # image_files = sorted(glob.glob(os.path.join(indir,"*.png")))
    image_files = sorted(glob.glob(os.path.join(indir, "images", "*.png")),key=lambda f: int(os.path.split(f)[1].rsplit(os.path.extsep, 1)[0].rsplit(None,1)[-1]))
    
    pbar = tqdm.tqdm(total=len(image_files)-1, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    num_events = 0

    (height,width,_)=cv2.imread(image_files[0]).shape
    device=torch.device("cuda")
    
    # timestamp_ns=counter*1e6/args.fps
    # timestamps=np.array([i/args.fps for i in range(len(image_files))])
    # timestamps_ns = (timestamps * 1e9).astype("int64")
    # timestamps_ns = torch.from_numpy(timestamps_ns).cuda()
    # no_kronecker_frames=int(timestamps[-1]*args.fps+1)
    kronecker_idx=3
    kronecker_latest_timestamp=int(kronecker_idx*1e9/args.fps)
    idx_accumulated=torch.zeros(size=[0],dtype=torch.int32,device=device)
    for image_file, timestamp_ns in zip(image_files, timestamps_ns):
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        log_image = np.log(image.astype("float32") / 255 + 1e-5)
        log_image = torch.from_numpy(log_image).cuda()

        sub_events = esim.forward(log_image, timestamp_ns)

        # for the first image, no events are generated, so this needs to be skipped
        if sub_events is None:
            continue

        if(timestamp_ns>=kronecker_latest_timestamp):
            # print("time to generate kronecker")
            #kroneckerdelta generation
            xs = sub_events['x'].to(torch.int32).to(device)
            ys = sub_events['y'].to(torch.int32).to(device)
            idx=(xs + (ys)* width)
            ts_contig=sub_events['t'].contiguous()
            last_event_idx=torch.searchsorted(ts_contig,kronecker_latest_timestamp,side='right').item()
            # print("last idx found at ",last_event_idx," among ",xs.shape[0]," events")
            # print(kronecker_latest_timestamp/1e9," ",kronecker_idx)
            idx_accumulated=torch.concatenate((idx_accumulated,idx[:last_event_idx]))
            kronecker_img = torch.zeros((height*width), dtype=torch.int16,device=device)
            ps=torch.ones((idx_accumulated.shape[0]),dtype=torch.int16,device=device)
            kronecker_img.index_add_(dim=0,index=idx_accumulated,source=ps)
            idx_accumulated=idx[last_event_idx:]



            # 90 percentile max
            nonzero_mask=kronecker_img != 0
            nonzero_voxel = kronecker_img[nonzero_mask]
            if nonzero_voxel.numel()>0:
                event_min=max(0,nonzero_voxel.min().item())
                scale=255/ max(1,(torch.quantile(nonzero_voxel.to(torch.float),0.9)-event_min))
            else:
                scale=255
            kronecker_img_out = torch.reshape( torch.clamp((kronecker_img*scale-event_min),0,255).to(torch.uint8) , (height, width))
            kronecker_img_out = kronecker_img_out.to(torch.device("cpu")).numpy()

            cv2.imwrite(os.path.join(outdir, "frame_%010d.png" % kronecker_idx), kronecker_img_out)
            # sub_events = {k: v.cpu() for k, v in sub_events.items()}    
            
            kronecker_idx+=1
            kronecker_latest_timestamp=int((kronecker_idx+1)*1e9/args.fps)
        else:
            # print("not time to generate kronecker")
            xs = sub_events['x'].to(torch.int32).to(device)
            ys = sub_events['y'].to(torch.int32).to(device)
            idx=(xs + (ys)* width)
            idx_accumulated=torch.concatenate((idx_accumulated,idx))
        # # do something with the events
        # np.savez(os.path.join(outdir, "%010d.npz" % counter), **sub_events)
        num_events += len(sub_events['t'])
        pbar.set_description(f"Num events generated: {num_events}")
        pbar.update(1)
    #   generate last image
    kronecker_img = torch.zeros((height*width), dtype=torch.int16,device=device)
    ps=torch.ones((idx_accumulated.shape[0]),dtype=torch.int16,device=device)
    kronecker_img.index_add_(dim=0,index=idx_accumulated,source=ps)

    # 90 percentile max
    nonzero_mask=kronecker_img != 0
    nonzero_voxel = kronecker_img[nonzero_mask]
    if nonzero_voxel.numel()>0:
        event_min=max(0,nonzero_voxel.min().item())
        scale=255/ max(1,(torch.quantile(nonzero_voxel.to(torch.float),0.9)-event_min))
    else:
        scale=255
    kronecker_img_out = torch.reshape( torch.clamp((kronecker_img*scale-event_min),0,255).to(torch.uint8) , (height, width))
    kronecker_img_out = kronecker_img_out.to(torch.device("cpu")).numpy()

    cv2.imwrite(os.path.join(outdir, "frame_%010d.png" % kronecker_idx), kronecker_img_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Generate events from a high frequency video stream""")
    parser.add_argument("--contrast_threshold_negative", "-cn", type=float, default=0.2)
    parser.add_argument("--contrast_threshold_positive", "-cp", type=float, default=0.2)
    parser.add_argument("--refractory_period_ns", "-rp", type=int, default=0)
    parser.add_argument("--input_dir", "-i", default="", required=True)
    # parser.add_argument("--output_dir", "-o", default="", required=True)
    parser.add_argument("--fps", "-f", default=30)
    args = parser.parse_args()


    print(f"Generating events with cn={args.contrast_threshold_negative}, cp={args.contrast_threshold_positive} and rp={args.refractory_period_ns}")

    # for folder_path, subdirs, files in os.walk(args.input_dir):
    #     if is_valid_dir(subdirs, files):
    output_folder=args.input_dir+f"_vid2e/images"
    print(output_folder)
    ensure_dir(output_folder)
    os.system(f"cp -r {args.input_dir}/../rgb_yolo_labels {args.input_dir}_vid2e/labels")
    process_dir(output_folder, args.input_dir, args)
