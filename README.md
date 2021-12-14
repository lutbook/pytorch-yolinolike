# pytorch-yolinolike
Simple implementation of YOLinO model in pytorch. (Not official )

This is simple implementation of YOLinO [1] structure.
Inputs and outputs are formatted for my purpose of the implementation.


General purpose of the model is detecting polyline.
Main concept is similar to populor YOLO models.
Segment of polylines are detected by grid cells. 
But detected line segment is only for the grid cell, not for entire image.


Getting full advantage of the model, further implemantation is needed.


[1]: Annika Meyer, Philipp Skudlik, Jan-Hendrik Pauls, Christoph Stiller, "YOLinO: Generic Single Shot Polyline Detection in Real Time", (2021), arXiv:2103.14420v1 [cs.CV]. https://arxiv.org/pdf/2103.14420v1.pdf
