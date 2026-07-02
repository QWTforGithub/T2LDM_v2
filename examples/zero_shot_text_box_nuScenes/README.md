I am very busy such that I do not have time to finish this section. <br/>
If I have enough time in the future, I will make up it.<br/>
Actually, it is very easy to implement this.<br/>
1. You first encode all texts in T2nuScenes++ by T5.
2. Then, you encode the input text by T5.
3. The L2 distance is used to find the matching index.
4. You can get the 3D box though the matching index.
5. Box-to-LiDAR generation is conducted.

It is very easy for you, right?
