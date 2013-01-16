topicmodeling
=============

Internal topic modeling library in scala

\/\/\/\/\/\/\/\/

After some time of development it was decided to rewrite it completely in python.

Here are the reasons for doing this:
  * JVM really eats a lot of memory and is not intended for technical computing. 
  * Existing math libraries for scala suck and are not simply supported by their developers. 
  * Pure java libraries just suck. In best case they use BLAS as backend while BLAS is really old and there are much more efficient libraries.
  * Despite the fact that it's relatively easy to parallelize and distribute computations in scala, it should be possible to do the same things in python with not much more effort
  * numpy in python is really fast and cool 
