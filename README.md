# Motion_Style_Infilling-pytorch-
Motion_Style_Infilling(pytorch) / GCT525 Motion Graphics Term project







기록.



Batch norm을 넣어서 학습이 안정되었다. 이전에 특정 시점에서 갑자기 값이 튀는 문제 발생.



AE는 infilling이 어느정도 잘 됨.
Decoder에 BN 빼도 어느정도 잘됨 -> 좀 가만히 있지를 못하는거 같기도? 흔들림. 떨리는거랑 조금 다르게


VAE로 하니까 결과가 떨림.


StyleTransfer
AdaIN은 일단 Non pretrained시 아주 이상한 결과가 나옴
하지만 확인한하고 갖다 붙여서 한거니까 다시 들여다볼 필요는 있음.

recon 1.0 일떄 (style loss x ) A-> A는 잘 되는편

하지만 Style 반 contetn 반이 명확하게 가이드 되지않음

recon weight넣어서 줄이니까 움직이지도 않음.

디노이징(빈칸 x , 노이즈만 첨가)은 VAE, AE 모두 잘 됨

GANbased는 일단 학습이 안됨 그냥 서있음->>오래 학습한 결과 그냥 reconsturction label 없이 B를 real로 넣어준거로는 스타일 전이가 되지 않음.

selfRef_sameWeights 뭔가 망가지진 않는데, 뭘 보고 배우는지 모르겠음 하나의 인풋 하나의 스타일로 매칭되는 결과는 아님.

baseline으로 motion blending한거를 사용해야 할수도 있음 . -> VAE로 가정 넣어서 latent space에서 affine transform ㄱㄱ

##blend-guided 된다!! 
과연 novelty를 어떻게 주장할 수 있는가? 걍 blend보고 배운건가? 근데 encode가 있긴해
GAN을 추가해줘야 novelty가 사려나? 



그리고 가운데 들어가는 프레임이 꼭 같은 위치일 필요 없다! -.>> random하게 주니까 style clip 포즈의 mean 으로 애매하게 되는 느낌도 있는거 같음.
-->>> differenet region을 잘라주면 그  refer clip의 mean pose 1개에 갔다가 오는느낌

static region이면 가운데거를 모방하는 느낌

-> GAN 추가 해줘도 잘 된다.  GAN 추가해도 differenet region, static region 경향성은 같음