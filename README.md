# Rock-Physics-GSA

Technical Implementation of the Generalized Singular Approximation (GSA) Method for Two-Component Rocks.

<h2>Summary of Completed Tasks</h2>
<ul>
  <li>Analysis of the fundamental GSA formula and its adaptation for two-component rock formations</li>
  <li>Decomposition of the GSA calculation process and a qualitative analysis of computation times for each step to identify the most problematic stages</li>
  <li>Planning a future computation scheme, taking into account the issues identified in the previous task</li>
  <li>Validation of obtained results using reference tabular data</li>
  <li>Evaluation of available technologies to reduce computation time, both for the entire program and the most resource-intensive steps</li>
  <li>Adaptation of code for JIT compilation usage: modifying various language constructs and breaking down existing functions to encapsulate JIT-executable logic</li>
</ul>

<h2>Theoretical Part</h2>
Within the framework of the GSA model, rock formations are approximated by a set of elliptical inclusions.

Each component of the rock corresponds to a group of inclusions with fixed parameters:
<ul>
  <li>Aspect ratio for the ellipsoid axes</li>
  <li>Orientation in the modeled volume</li>
  <li>Elasticity tensor</li>
</ul>

The model also takes into account the degree of connectivity between components.

<h3>Main Formula</h3>
In general, the effective elastic tensor for a multi-component rock is calculated using the following formula:
<img src="https://github.com/Fedor4096/Rock-Physics-GSA/assets/108585151/ccb5643f-4778-4ffc-9793-9ea6bb81424a" height="120" vspace="10"><br>
<i>*In the formula, tensor operations are required: multiplication and finding the inverse tensor.</i><br><br>

Within this work, the functionality has been implemented only for the <strong>two-component</strong> case, and the ability to change the inclusion <strong>orientation</strong> in space is not available (horizontally fixed).

The main computational challenge lies in finding the tensor components <strong>g</strong> that describe the interaction of the i-th inclusion in the rock with its nearest neighbors.

<h3>Calculation of the Tensor g (Second Derivative of the Green's Function)</h3>
Calculating the tensor <strong>g</strong> is a complex, multi-stage process. It requires the use of an auxiliary tensor <strong>a</strong> and the application of numerical integration methods.

The process of calculating <strong>g</strong> involves the following steps:
<ol>
  <li>Calculation of components of the auxiliary non-symmetric tensor <strong>a<sub>not sym</sub></strong>:</li>
  <img src="https://github.com/Fedor4096/Rock-Physics-GSA/assets/108585151/28cae17e-b819-4b26-b6e7-9835a0228764" height="70" vspace="30">, where
  <img src="https://github.com/Fedor4096/Rock-Physics-GSA/assets/108585151/9b73ac8d-7960-486d-b6f8-7c70dbb42666" height="50" vspace="30"> and
  <img src="https://github.com/Fedor4096/Rock-Physics-GSA/assets/108585151/439c47ed-bd82-44b9-8cc3-ba4297fa70ca" height="50" vspace="30">

  <li>Symmetrization of the obtained tensor <strong>a<sub>not sym</sub></strong>:</li>
  <img src="https://github.com/Fedor4096/Rock-Physics-GSA/assets/108585151/5c555a87-96a5-4ef2-9012-bf419826c222" height="60" vspace="30">
  
  <li>Obtaining the tensor <strong>g</strong> by reassigning components of the symmetrized <strong>a<sub>sym</sub></strong>:</li>
  <img src="https://github.com/Fedor4096/Rock-Physics-GSA/assets/108585151/bb17222f-aad9-421c-ae75-115152ba497f" height="70" vspace="30">
</ol>

<h3>Numerical Integration</h3>
To obtain the components of the auxiliary non-symmetric tensor <strong>a<sub>not sym</sub></strong>, it is necessary to calculate an integral over a surface defined by the sub-integrand function.

In this work, the integral was computed using a method with <strong>predefined</strong> integration nodes based on the following formula:
<img src="https://github.com/Fedor4096/Rock-Physics-GSA/assets/108585151/57fe586e-420c-4aee-8589-2204d0f499ae" height="70" vspace="30">

It is important to note that selecting the right integration nodes can be a challenging task, as it requires a detailed understanding of the behavior of the sub-integrand functions.



<h2>Practical Part</h2>
The GSA model belongs to the class of algorithms where calculation speed is critically important.

In the context of this project, the following tasks were carried out:
<ul>
  <li><strong>Theoretical Level</strong> - Identifying the most optimal order of variable calculations</li>
  <li><strong>Technical Level</strong> - Using JIT compilation methods for comprehensive program optimization</li>
</ul>

The most significant reduction in computation time was achieved through the application of JIT compilation. As a result, calculation speed was improved by a factor of three.

It is important to note that during the first run of the algorithm, a significant amount of time is spent compiling the required program logic into machine code.

However, in most cases, the GSA algorithm is used sequentially and repeatedly. Therefore, the time spent solely on the initial iteration for compiling functions into machine code makes a negligible contribution when there are a large number of iterations.

Thus, the use of the Python language in conjunction with JIT compilation in this case allows one to approach the results obtained with compiled languages such as Fortran, C, or Rust. Unlike the stock Python, which is on average 500 times slower than compiled counterparts, the combination of Python and JIT reduces the gap to 150 times on the first iteration and subsequently executes the program 40-50 times slower.



