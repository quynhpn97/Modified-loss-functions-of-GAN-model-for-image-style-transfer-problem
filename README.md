# Modified-loss-functions-of-GAN-model-for-image-style-transfer-problem

Image style transfer is an interesting research in computer vision. The combination of artificial intelligence and art makes this technique highly concerned in the relevant technical fields and art fields. Recent studies have focused on improving the relationship between object and style. There are many proposals for the loss function to improve this.

Specifically, I have summarized some proposals for loss function and deployed with some models including CycleGAN [[2]](#2), GAN model with dual-consistency loss [[1]](#1).

Details on: 

Results and comparison:
<table>
  <tr>
    <td> Content
    <td> Dual Consistency loss GAN
    <td> Cycle GAN
    <td> Style
  </tr>
  
  <tr>
    <td> 
    <td> <img src="images/md1.png" alt = "1">
    <td> <img src="images/mc1.png" alt = "1">
    <td> <img src="images/monet.png"  alt="1" >
  </tr>
  
  <tr>
    <td> <img src="images/town2.png"  alt="1" >
  </tr>
  
  <tr>
    <td> 
    <td><img src="images/cd1.png" alt="2" >
    <td><img src="images/cc1.png" alt="2" >
    <td><img src="images/ceznna.png" alt="2" >
  </tr>
</table>


<table>
  <tr>
    <td> Content 
    <td> Dual Consistency loss GAN
    <td> Cycle GAN
    <td> Style
  </tr>
  
  <tr>
    <td> 
    <td> <img src="images/md2.png" alt = "1">
    <td> <img src="images/mc2.png" alt = "1">
    <td> <img src="images/monet.png"  alt="1" >
  </tr>
  
  <tr>
    <td> <img src="images/town.png"  alt="1" >
  </tr>
  
  <tr>
    <td>
    <td><img src="images/cd2.png" alt="2" >
    <td><img src="images/cc2.png" alt="2" >
    <td><img src="images/ceznna.png" alt="2" >
  </tr>
</table>

## References
<a id="1">[1]</a> 
Zhuoqi Ma, Jie Li, Nannan Wang, Xinbo Gao,
Semantic-related image style transfer with dual-consistency loss.,
Neurocomputing,
Volume 406,
2020,
Pages 135-149,
ISSN 0925-2312,
https://doi.org/10.1016/j.neucom.2020.04.027.

<a id="2">[2]</a> 
Jun-Yan Zhu and T. Park and Phillip Isola and Alexei A. Efros,
Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks,
2017 IEEE International Conference on Computer Vision (ICCV),
2017,
Pages 2242-2251.
