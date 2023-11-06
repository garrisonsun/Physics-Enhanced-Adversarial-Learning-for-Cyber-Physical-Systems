
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">Securing Cyber-Physical Systems:
Physics-Enhanced Adversarial Learning for
Autonomous Platoons</h1>
<p align="center">
  
  <details open="open">
  <summary>Table of Contents</summary>
  <ol>
      <li><a href="#about-the-project">About The Project</a></li>
      </ul>
    </li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#simulation">Simulation</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#cite-this-work">Cite this work</a></li>
    <li><a href="#license">License</a></li>
   </ol> 
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This work combines cyber-physical system characteristics with DL to develop a hybrid attack detection system. Using knowledge from both physical dynamics and data, we defend against both cyber-physical attacks and adversarial attacks. 
<!-- GETTING STARTED -->
## Getting Started
- The evaluation results presented in the paper is prepared with the Python file in the folder 'Evaluation'. The testing results are in '.csv' format, which can be easily interpreted. 
- Folder 'Data' contains the complete dataset for training and testing.
  1. Main training data is contained in the folder 'train_raw' and 'leader_raw' contains the information of attack initiation time etc. Only normal driving behaviour is used in model training and attack behaviour is used in adversarial retraining. The complete dataset also contains other vehicle information that is not used to train the model.
  2. Three scenarios are considered to generate data: 
      1. normal driving scenario with different traffic conditions; 
      2. repeating the process of accelerating from 0 to 70/90 km/h then decelerating to 0 km/h; 
      3. randomly change speed between 70-90 km/h.
- The PDF file 'SupplementaryMaterial' contains other implementation details to back up the paper.
## Simulation
### Defend agaisnt Conventional Cyber-physical Attacks
https://user-images.githubusercontent.com/50089203/174946082-b3a46cc9-fc92-48b2-8065-86ae10bca08b.mp4
### Defend agaisnt Adversarially-masked Cyber-physical Attacks
https://user-images.githubusercontent.com/50089203/174946606-ea2f37f6-bd54-4673-a65b-36fdc6bd7a92.mp4

## Authors
- [Guoxin Sun](https://electrical.eng.unimelb.edu.au/people/research-students)

- [Tansu Alpcan](https://findanexpert.unimelb.edu.au/profile/425318-tansu-alpcan)

- [Ben Rubinstein](https://findanexpert.unimelb.edu.au/profile/20074-ben-rubinstein)
  
- [Seyit Camtepe](https://people.csiro.au/C/S/Seyit-Camtepe)

## Cite this work
If you find this work useful, please consider to cite the following reference [paper](https://garrisonsun.github.io/End-to-end-atttack-detection-and-mitigation-framework/):
```
@inproceedings{sun2022securing,
  title={Securing Cyber-Physical Systems: Physics-Enhanced Adversarial Learning for Autonomous Platoons},
  author={Sun, Guoxin and Alpcan, Tansu and Rubinstein, Benjamin IP and Camtepe, Seyit},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={269--285},
  year={2022},
  organization={Springer}
}
```
<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

