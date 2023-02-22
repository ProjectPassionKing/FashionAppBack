# FashionBack

Python 기반의 FashionApp 프로젝트의 백엔드 repository  
FashionApp의 안드로이드 어플리케이션의 경우, 별도 repository를 참고하십시오. (https://github.com/soojlee0106/FashionApp)  

## Model
YOLO v8을 이용한 bounding box, 상의 하의 판정 (https://github.com/ultralytics/ultralytics)    
Vision Transformer를 이용한 의류 별 classification 

## Version Info
- Python: 3.8

## Run docker

```sh
docker run -p 5000:5000 --rm flaskwebapp
```

## Live Server Azure 주소
```sh
https://fashionapp.azurewebsites.net/
```
