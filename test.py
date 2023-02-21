import torch


class Detector():
    def __init__(self) -> None:
        self.model = torch.hub.load(
            'ultralytics/yolov5', "yolov5n", pretrained=True, trust_repo=True)

    def detect(self, frame):
        result = self.model(frame)
        result_table = result.pandas().xyxy[0]
        process_time = sum(result.t)
        confidence = result_table["confidence"]
        x_min = result_table["xmin"]
        y_min = result_table["ymin"]
        x_max = result_table["xmax"]
        y_max = result_table["ymax"]
        object_name = result_table["name"]
        print(type(confidence.values))
        print(confidence.values.tolist())
        print("=" * 20)
        print(x_min)
        print(type(result_table))
        for index, row in result_table.iterrows():
            print(row)
            print(type(row))
            print(row.values.tolist())


if __name__ == "__main__":
    detector = Detector()
    detector.detect('https://ultralytics.com/images/zidane.jpg')
