import onnxruntime as ort
import numpy as np
import gc
import cv2
import time
import win32api
import win32con
import pandas as pd
import torch
from utils.general import (cv2, non_max_suppression, xyxy2xywh)

# Configuration imports
from config import aaMovementAmp, useMask, maskHeight, maskWidth, aaQuitKey, confidence, headshot_mode, cpsDisplay, visuals, onnxChoice, centerOfScreen
import gameSelection

def main():
    camera, cWidth, cHeight = gameSelection.gameSelection()

    # --- TUNING ---
    fov_radius = 80 # 0.5 inch zone
    # --------------

    count = 0
    sTime = time.time()
    
    # Targeting persistence variables
    locked_target_coord = None 

    onnxProvider = ""
    if onnxChoice == 1: onnxProvider = "CPUExecutionProvider"
    elif onnxChoice == 2: onnxProvider = "DmlExecutionProvider"
    elif onnxChoice == 3:
        import cupy as cp
        onnxProvider = "CUDAExecutionProvider"

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_sess = ort.InferenceSession('yolov5m320Half.onnx', sess_options=so, providers=[onnxProvider])

    COLORS = np.random.uniform(0, 255, size=(1500, 3))

    while win32api.GetAsyncKeyState(ord(aaQuitKey)) == 0:
        npImg = np.array(camera.get_latest_frame())

        from config import maskSide 
        if useMask:
            maskSide = maskSide.lower()
            if maskSide == "right": npImg[-maskHeight:, -maskWidth:, :] = 0
            elif maskSide == "left": npImg[-maskHeight:, :maskWidth, :] = 0

        # Normalization
        if onnxChoice == 3:
            im = torch.from_numpy(npImg).to('cuda')
            if im.shape[2] == 4: im = im[:, :, :3]
            im = torch.movedim(im, 2, 0).half() / 255
            if len(im.shape) == 3: im = im[None]
            outputs = ort_sess.run(None, {'images': cp.asnumpy(im)})
        else:
            im = np.array([npImg])
            if im.shape[3] == 4: im = im[:, :, :, :3]
            im = (im / 255).astype(np.half)
            im = np.moveaxis(im, 3, 1)
            outputs = ort_sess.run(None, {'images': np.array(im)})

        im_output = torch.from_numpy(outputs[0]).to('cpu')
        pred = non_max_suppression(im_output, confidence, confidence, 0, False, max_det=10)

        targets = []
        for i, det in enumerate(pred):
            gn = torch.tensor(im_output.shape)[[0, 0, 0, 0]]
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    targets.append((xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() + [float(conf)])

        df = pd.DataFrame(targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence"])

        # STICKY TARGETING LOGIC
        final_target = None
        lmb_pressed = win32api.GetAsyncKeyState(0x01) != 0
        master_on = win32api.GetKeyState(0x14) # CapsLock

        if len(df) > 0:
            # 1. Calculate distance from crosshair for all
            df["dist_from_center"] = np.sqrt((df.current_mid_x - cWidth)**2 + (df.current_mid_y - cHeight)**2)
            
            # 2. If we were already tracking someone and still holding LMB, find them again
            if lmb_pressed and locked_target_coord is not None:
                # Find the detection closest to where the target was in the last frame
                df["dist_from_last"] = np.sqrt((df.current_mid_x - locked_target_coord[0])**2 + (df.current_mid_y - locked_target_coord[1])**2)
                # Filter for detections within a reasonable "movement range" of the last frame
                potential_targets = df[df["dist_from_last"] < 50] 
                if not potential_targets.empty:
                    final_target = potential_targets.sort_values("dist_from_last").iloc[0]

            # 3. If no locked target, grab the one closest to crosshair
            if final_target is None:
                closest_to_center = df.sort_values("dist_from_center").iloc[0]
                if closest_to_center["dist_from_center"] <= fov_radius:
                    final_target = closest_to_center

        # EXECUTE MOVEMENT
        if final_target is not None:
            # Update the lock-on coordinate
            locked_target_coord = [final_target.current_mid_x, final_target.current_mid_y]
            
            if lmb_pressed and master_on:
                # Calculate movement
                box_h = final_target.height
                offset = box_h * (0.33 if headshot_mode else 0.25)
                
                # Smoothed movement
                move_x = int((final_target.current_mid_x - cWidth) * aaMovementAmp)
                move_y = int(((final_target.current_mid_y - offset) - cHeight) * aaMovementAmp)
                
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)
        # else:
        #     # Reset lock if no targets are found
        #     locked_target_coord = None

        # VISUALS
        if visuals:
            cv2.circle(npImg, (int(cWidth), int(cHeight)), fov_radius, (0, 255, 255), 1)
            for i in range(len(df)):
                t = df.iloc[i]
                cv2.rectangle(npImg, (int(t.current_mid_x - t.width/2), int(t.current_mid_y - t.height/2)), 
                              (int(t.current_mid_x + t.width/2), int(t.current_mid_y + t.height/2)), (0, 255, 0), 2)
            cv2.imshow('Live Feed', npImg)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break

        # Timing
        count += 1
        if (time.time() - sTime) > 1:
            if cpsDisplay: print(f"CPS: {count}")
            count, sTime = 0, time.time()

    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()