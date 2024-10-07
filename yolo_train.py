from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    # Load a model
    model = YOLO("./weights/yolo11n.pt")

    # Train the model
    train_results = model.train(
        data="yolo_dataset.yaml",  # path to dataset YAML
        epochs=350,  # number of training epochs
        imgsz=1024,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=16,
        workers=4,
        patience=150,
    )

    # model = YOLO('./runs/detect/train3/weights/last.pt')
    # train_results = model.train(resume=True)

    # Evaluate model performance on the validation set
    # metrics = model.val()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model
    print(path)