# resnet_pytorch
This code implements a version of the common convolutional neural network architecture **ResNet** and the necessary workflow surrounding deep learning algorithms using the opensource library **PyTorch**.
I use my implementation to detect defects on solar cells. To this end, a dataset containing images of solar cells was provided by the university. In this work, we focus on two different types of defects:
1. **Cracks**: The size of cracks may range from very small cracks to large cracks that cover the whole cell. In most cases, the performance of the cell is unaffected by this type of defect, since connectivity of the cracked area is preserved.
2. **Inactive regions**: These regions are mainly caused by cracks. It can happen when a part of the cell becomes disconnected and therefore does not contribute to the power production. Hence, the cell performance is decreased.
Of course, the two defect types are related, since an inactive region is often caused by cracks. However, we treat them as independent and only classify a cell as cracked if the cracks are visible.
This is essentially a **multi-label classification** where the labels of each solar cell image are two numbers indicating if the solar cell shows a "crack" and if the solar cell can be considered inactive.
