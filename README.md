# histogram_matching
2023.04.23 ~ 27 ETRI 소셜로봇연구실 진행사항
 
## 히스토그램 매칭이란?

이미지의 색 분포를 목표 이미지의 색 분포와 비슷하게 하는 알고리즘이다.

![1](https://github.com/Seungeun-Han/histogram_matching/assets/101082685/22f210fd-ad54-4c4e-b370-29930ab8c4d0)

![2](https://github.com/Seungeun-Han/histogram_matching/assets/101082685/7606f7f5-603b-47ec-a041-341c59be26e3)

출처: https://swprog.tistory.com/entry/Histogram-Matching-Color-mapping-Color-Transfer


### Example
초록 계열의 색이 많은 숲 이미지를 파란 계열의 색이 많은 바다 이미지의 히스토그램을 따르도록 매칭시키는 예시이다.

- source image

![forest](https://github.com/Seungeun-Han/histogram_matching/assets/101082685/42d50d5f-c6d6-4620-9c99-eeb40634ef43)


- target image

![sea](https://github.com/Seungeun-Han/histogram_matching/assets/101082685/cc3aae70-b257-4e41-a7e0-18b0f0df03e7)

- output

바다 색 분포를 가진 숲으로 변환된다.

![out](https://github.com/Seungeun-Han/histogram_matching/assets/101082685/72119d4f-a929-4c2c-ab45-4de27733bcf8)


<hr>


## 우리가 하려는 것

### face parsing → mask inpainting 과정에서 논리적 문제

face parsing을 통해 mask 영역을 segmentation 하고, 해당 영역을 inpainting 모델로 복원하는 연구를 진행하였다.

하지만 inpainting 모델의 복원 결과 코나 입 부분의 shape은 잘 복원 되지만, 피부톤이 blue가 강조되며 복원되는 것을 볼 수 있었다.

![image](https://github.com/Seungeun-Han/histogram_matching/assets/101082685/f253c1f4-5994-4a0f-8409-2fe9a7852108)


그래서 inpainting 된 영역의 히스토그램을 피부 영역의 히스토그램으로 매칭시켜 파란 부분 보정하기 위해 히스토그램 매칭을 사용하였다.


### Example

- source image - inpainting 영역

![image](https://github.com/Seungeun-Han/histogram_matching/assets/101082685/654f280c-6c30-4ba6-a404-7756fffcbf1d)

- target image - skin 영역

![image](https://github.com/Seungeun-Han/histogram_matching/assets/101082685/86a136a4-852d-4de2-b863-e3dfcc2f7f18)

- output

각 채널 별로 히스토그램 매칭이 잘 되었는지 확인하기 위해 히스토그램을 출력하였다.
왼쪽은 원본 영상의 히스토그램, 중간은 타겟 영상의 히스토그램, 오른쪽은 결과 영상의 히스토그램이다.

- b channel

![KakaoTalk_20230525_200944778_01](https://github.com/Seungeun-Han/histogram_matching/assets/101082685/2ab767ab-51bf-4539-9eef-74a995e89370)

- g channel

![KakaoTalk_20230525_200944778_02](https://github.com/Seungeun-Han/histogram_matching/assets/101082685/b429d09a-1c66-4c33-8c1a-1ea9a3411d8a)

- r channel

![KakaoTalk_20230525_200944778_03](https://github.com/Seungeun-Han/histogram_matching/assets/101082685/8faa8a4f-84c3-4f08-8bb8-6f9c00578593)


원본 영상과 히스토그램이 비슷해진것을 볼 수 있다.


## How to use?
이 repository를 clone하고 histogram_matching_color_image.py 코드를 실행하면 된다.

```
mkdir ${YOUR_PROJECT_NAME}
cd ${YOUR_PROJECT_NAME}
git clone https://github.com/google-research/deeplab2.gitmkdir](https://github.com/Seungeun-Han/histogram_matching.git
```


