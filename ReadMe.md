# **Solving Partial Differential Equations using Deep Learning**
Neural Operators are a relatively new method of approximating the solution of complex PDEs which occur in physics. This project is meant to be an implementation of a few of these methods to discern how exactly they work.


_PDEBench is a benchmark introduced to test the performance of these techniques. It also provides datasets which I will be using. The PDEBench datasets are large. I need to figure out how to deal with data on the scale of many hundreds of gigabytes._ 




A well documented technique of solving PDE's are PINNs. If not included in the main directory, it warrants an implementation. 

The neural operator architectures encountered and dealt with are: 
1. Deep Operator Networks (DeepONet)
2. Fourier Neural Operators (FNO)
3. Geometrically Aware Neural Operators (geoFNO)
4. General Adversarial Neural Operators (GANO)
5. Physics Informed Neural Operator (PINO)

Notes: 
    As the PDEBench datasets provided are large, I will be focusing on 2D darcy flow and 1DS advection to begin with.  

## **References**

* Neural Operator Paper
* FNO paper
* PDEBench paper
* PINN paper
