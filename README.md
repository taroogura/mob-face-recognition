# mob-face-recognition

mobilenet face recognition

## Install

```
pip install git+https://github.com/taroogura/mob-face-recognition
```

## demo

### download test images

```bash
curl http://vis-www.cs.umass.edu/lfw/images/Aaron_Peirsol/Aaron_Peirsol_0001.jpg -o ap1.jpg
curl http://vis-www.cs.umass.edu/lfw/images/Aaron_Peirsol/Aaron_Peirsol_0002.jpg -o ap2.jpg
curl http://vis-www.cs.umass.edu/lfw/images/Aaron_Sorkin/Aaron_Sorkin_0001.jpg -o as1.jpg
```

### extract features and match

```python
import cv2
from mob_face_recognition.mobface import Mobface, compare

mobface = Mobface()
conf_ap1, feat_ap1 = mobface.extract(cv2.imread('ap1.jpg'))
conf_ap2, feat_ap2 = mobface.extract(cv2.imread('ap2.jpg'))
conf_as1, feat_as1 = mobface.extract(cv2.imread('as1.jpg'))

mate_score = compare(feat_ap1, feat_ap2)
nonmate_score1 = compare(feat_as1, feat_ap1)
nonmate_score2 = compare(feat_as1, feat_ap2)
print(mate_score, nonmate_score1, nonmate_score2)   #  0.7762308, -0.12886877, -0.09556435
```

