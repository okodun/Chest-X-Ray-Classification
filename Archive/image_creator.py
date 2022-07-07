from grad_cam import GradCam

gc = GradCam("new_save_at_100.h5")
gc.create(
    "/home/felix/Documents/University/SS2022/ML4B/Data Set Chest X-Ray/chest_xray/train/NORMAL/NORMAL2-IM-1385-0001.jpeg",
    cam_path="Pictures/Evaluation Quiz/wrong_classification.jpeg")
gc.annotate("Pictures/Evaluation Quiz/cats_new.jpeg")
