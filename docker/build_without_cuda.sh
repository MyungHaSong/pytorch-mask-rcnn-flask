
echo "build 1"
cd nms
python build.py
cd ..

echo "build 2"
cd roialign/roi_align/
python build.py
cd ../../

echo "clone cocoapi"
git clone https://github.com/cocodataset/cocoapi.git
echo "build/settings cocoapi"
cd cocoapi/PythonAPI
make
cd ../..
ln -s cocoapi/PythonAPI/pycocotools pycocotools