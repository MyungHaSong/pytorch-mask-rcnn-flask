
echo "build 1"
cd nms/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=$GPU_Arch
cd ../../
#sed 's/torch.cuda.is_available()/True/g' build.py > output.txt 
#rm build.py 
#mv output.txt build.py
python build.py

cd ..

echo "build 2"
cd roialign/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=$GPU_Arch
cd ../../
#sed 's/torch.cuda.is_available()/True/g' build.py > output.txt
#rm build.py
#mv output.txt build.py
python build.py
cd ../../

echo "clone cocoapi"
git clone https://github.com/cocodataset/cocoapi.git
echo "build/settings cocoapi"
cd cocoapi/PythonAPI
make
cd ../..
ln -s cocoapi/PythonAPI/pycocotools pycocotools