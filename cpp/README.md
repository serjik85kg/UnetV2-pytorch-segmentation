## C++ torchscript project
This is the raw template C++ project for including pytorch model inference in different C++ projects.  
This project uses special zip weights for torchscript module **model_unet_traced.pt** which we can take from *python/utils/conversions/trace_model_unet.py*. 

### Requirements.
 - C++11 and later  
 - Opencv 3 and later
 - libtorch 1.5.1 and later  

Instructions for libtorch installation in Visual Studio:  
https://medium.com/@boonboontongbuasirilai/building-pytorch-c-integration-libtorch-with-ms-visual-studio-2017-44281f9921ea

## TO DO:
1) Add CUDA configuration (and test output weights from *trace_model_unet.py*).  
2) Delete deprecated std::pair<Mat, Mat> input type and use universal container for inputs.  
3) Add Special Inference class.  
4) Add command prompt inputs.  
