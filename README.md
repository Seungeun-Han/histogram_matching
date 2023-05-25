# histogram_matching
 
## 히스토그램 매칭이란?

이미지의 색 분포를 목표 이미지의 색 분포와 비슷하게 하는 알고리즘이다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/aaed350f-acd4-492d-aa3f-6ccf263fb9ca/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/531aa7a1-c2ba-46aa-94f5-00e8a8b290b2/Untitled.png)

### Example

- input

![forest.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fae80b68-c1af-4ef9-9053-28a9ac4fedc5/forest.png)

- target
- 
![sea.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5393ef37-6db8-49ec-b112-18331184f0c6/sea.png)

- output

바다 색 분포를 가진 숲으로 변환된다.

![out.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f2409adc-4553-4906-b5a6-d00765c12096/out.png)


## 우리가 하려는 것

### face parsing → mask inpainting 문제

face parsing을 통해 mask 영역을 segmentation 하고, 해당 영역을 inpainting 모델로 복원하는 연구를 진행하였다.

하지만 inpainting 모델의 복원 결과 코나 입 부분의 shape은 잘 복원 되지만, 피부톤이 blue가 강조되며 복원되는 것을 볼 수 있었다.

![000001-004.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7fdca062-2a6a-4569-aaf0-14dd379e62db/000001-004.jpg)

![000001-005.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/544bf2d8-094a-455c-a21a-f0a3aa96275f/000001-005.jpg)

![000028-004.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/68fcb35d-cf86-4660-9b48-36a807fe5616/000028-004.jpg)

![000016-005.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/efa96fe5-d158-40c0-9f7c-a458ae6c16d2/000016-005.jpg)

그래서 inpainting 된 영역의 히스토그램을 피부 영역의 히스토그램으로 매칭시켜 파란 부분 보정하기 위해 히스토그램 매칭을 사용하였다.

### Example

- input - inpainting 영역 = source image

![000001-004.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bb4e2e34-8915-4162-a211-72d2490b74c3/000001-004.jpg)

![000001-005.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/69a7f7c2-a254-415a-af64-bc3fc5707213/000001-005.jpg)

![000028-004.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5204d3b6-aea6-451a-ba8c-ac0db40e1780/000028-004.jpg)

![000016-005.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/771a3766-f114-4823-99bb-b36063b18274/000016-005.jpg)

- target - skin 영역 = target image

![000001-004.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5e9aef8b-48fd-4bde-8478-c069ac2e83a8/000001-004.jpg)

![000001-005.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8083eec3-8a73-4c75-9463-7966dac683d5/000001-005.jpg)

![000028-004.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3ecfb762-3e47-4cff-a55b-b1ee80907cbc/000028-004.jpg)

![000016-005.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b1f87ba6-3e83-4f01-8758-edce772bc930/000016-005.jpg)


