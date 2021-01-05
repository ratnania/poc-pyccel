# Loop transformation à la LOOPY

In this POC (Proof Of Concept), we invistigate the transformation of loops using Pyccel, as proposed in [LOOPY](https://github.com/inducer/loopy).
You need first to read the following [tutorial of LOOPY](https://github.com/inducer/loopy), before moving forward.

Such techniques are needed in order to perform [loop tiling](https://en.wikipedia.org/wiki/Loop_nest_optimization)

## Examples

In the sequel, we shall consider the following 3 examples

### A Loop over a 1 rank array

```python
@types('int', 'int[:]')
def f1(n, xs):
    for i in range(n):
        xs[i] = 2
```

### A Loop over a 2 rank array

```python
@types('int[:,:]')
def f2(xs):
    from numpy import shape
    n,m = shape(xs)
    for i in range(n):
        for j in range(m):
            xs[i,j] = 2
```

### A Loop over a 3 rank array

```python
@types('int[:,:,:]')
def f3(xs):
    from numpy import shape
    n,m,p = shape(xs)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                xs[i,j,k] = 2 
```

## Loop transformations

Before, showing how it is done using Pyccel, we will first show what the expected code will be.

In this POC, two techniques were implemented

* Loop Splitting: Splits a loop over an index (**i**) into two nested loops over an *outer* and *inner* index, with the inner loop having a fixed size (typically small)
* Loop Splitting + Unrolling of the inner loop

> **Note**: when splitting nested loops, the inner loops must be gathered at the nested level.

### Loop Splitting

By splitting the loop inside **f1** (index = i) with an inner size equal to 4, we shall get
```python
@types("int", "int[:]")
def f1(n, xs):
    for outer_i in range(0, (3 + n)//4, 1):
        for inner_i in range(0, 4, 1):
            xs[inner_i + 4 * outer_i] = 2
```

By splitting the loops inside **f2** (indices = i,j) with inner sizes equal to (2,4), we shall get

```python
@types("int[:,:]")
def f2(xs):
    from numpy import shape
    n,m = shape(xs)
    for outer_i in range(0, (1 + n)//2, 1):
        for outer_j in range(0, (3 + m)//4, 1):
            for inner_i in range(0, 2, 1):
                for inner_j in range(0, 4, 1):
                    xs[inner_i + 2 * outer_i,inner_j + 4 * outer_j] = 2 
```

By splitting the loops inside **f3** (indices = i,j,k) with inner sizes equal to (2,3,4), we shall get

```python
@types("int[:,:,:]")
def f3(xs):
    from numpy import shape
    n,m,p = shape(xs)
    for outer_i in range(0, (1 + n)//2, 1):
        for outer_j in range(0, (2 + m)//3, 1):
            for outer_k in range(0, (3 + p)//4, 1):
                for inner_i in range(0, 2, 1):
                    for inner_j in range(0, 3, 1):
                        for inner_k in range(0, 4, 1):
                            xs[inner_i + 2 * outer_i, inner_j + 3 * outer_j,inner_k + 4 * outer_k] = 2 
```

### Loop Splitting and unrolling

By splitting the loop inside **f1** (index = i) with an inner size equal to 4 and unrolling the inner loop, we shall get

```python
@types("int", "int[:]")
def f1(n, xs):
    for outer_i in range(0, (3 + n)//4, 1):
        xs[0 + 4 * outer_i] = 2
        xs[1 + 4 * outer_i] = 2
        xs[2 + 4 * outer_i] = 2
        xs[3 + 4 * outer_i] = 2
```

By splitting the loops inside **f2** (indices = i,j) with inner sizes equal to (2,4) and unrolling the inner loops, we shall get

```python
@types("int[:,:]")
def f2(xs):
    from numpy import shape
    n,m = shape(xs)
    for outer_i in range(0, (1 + n)//2, 1):
        for outer_j in range(0, (3 + m)//4, 1):
            xs[0 + 2 * outer_i,0 + 4 * outer_j] = 2
            xs[1 + 2 * outer_i,0 + 4 * outer_j] = 2
            xs[0 + 2 * outer_i,1 + 4 * outer_j] = 2
            xs[1 + 2 * outer_i,1 + 4 * outer_j] = 2
            xs[0 + 2 * outer_i,2 + 4 * outer_j] = 2
            xs[1 + 2 * outer_i,2 + 4 * outer_j] = 2
            xs[0 + 2 * outer_i,3 + 4 * outer_j] = 2
            xs[1 + 2 * outer_i,3 + 4 * outer_j] = 2
```

By splitting the loops inside **f3** (indices = i,j,k) with inner sizes equal to (2,3,4) and unrolling the inner loops, we shall get

```python
@types("int[:,:,:]")
def f3(xs):
    from numpy import shape
    n,m,p = shape(xs)
    for outer_i in range(0, (1 + n)//2, 1):
        for outer_j in range(0, (2 + m)//3, 1):
            for outer_k in range(0, (3 + p)//4, 1):
                xs[0 + 2 * outer_i,0 + 3 * outer_j,0 + 4 * outer_k] = 2
                xs[1 + 2 * outer_i,0 + 3 * outer_j,0 + 4 * outer_k] = 2
                xs[0 + 2 * outer_i,1 + 3 * outer_j,0 + 4 * outer_k] = 2
                xs[1 + 2 * outer_i,1 + 3 * outer_j,0 + 4 * outer_k] = 2
                xs[0 + 2 * outer_i,2 + 3 * outer_j,0 + 4 * outer_k] = 2
                xs[1 + 2 * outer_i,2 + 3 * outer_j,0 + 4 * outer_k] = 2
                xs[0 + 2 * outer_i,0 + 3 * outer_j,1 + 4 * outer_k] = 2
                xs[1 + 2 * outer_i,0 + 3 * outer_j,1 + 4 * outer_k] = 2
                xs[0 + 2 * outer_i,1 + 3 * outer_j,1 + 4 * outer_k] = 2
                xs[1 + 2 * outer_i,1 + 3 * outer_j,1 + 4 * outer_k] = 2
                xs[0 + 2 * outer_i,2 + 3 * outer_j,1 + 4 * outer_k] = 2
                xs[1 + 2 * outer_i,2 + 3 * outer_j,1 + 4 * outer_k] = 2
                xs[0 + 2 * outer_i,0 + 3 * outer_j,2 + 4 * outer_k] = 2
                xs[1 + 2 * outer_i,0 + 3 * outer_j,2 + 4 * outer_k] = 2
                xs[0 + 2 * outer_i,1 + 3 * outer_j,2 + 4 * outer_k] = 2
                xs[1 + 2 * outer_i,1 + 3 * outer_j,2 + 4 * outer_k] = 2
                xs[0 + 2 * outer_i,2 + 3 * outer_j,2 + 4 * outer_k] = 2
                xs[1 + 2 * outer_i,2 + 3 * outer_j,2 + 4 * outer_k] = 2
                xs[0 + 2 * outer_i,0 + 3 * outer_j,3 + 4 * outer_k] = 2
                xs[1 + 2 * outer_i,0 + 3 * outer_j,3 + 4 * outer_k] = 2
                xs[0 + 2 * outer_i,1 + 3 * outer_j,3 + 4 * outer_k] = 2
                xs[1 + 2 * outer_i,1 + 3 * outer_j,3 + 4 * outer_k] = 2
                xs[0 + 2 * outer_i,2 + 3 * outer_j,3 + 4 * outer_k] = 2
                xs[1 + 2 * outer_i,2 + 3 * outer_j,3 + 4 * outer_k] = 2
```

## Loop transformations using Pyccel

* A class **Transform** has been implemented. For the moment, it takes a filename and the **gather** boolean.
* We shall onlly perform code transformations when the semantic stage passes
* One of the consequences of the last point is that loops over **zip** or other iterables, are already treated and we get loops over **PythonRange**, which is assumed in the loop transformations
* We don't need to convert the code into Python in order to call **Transform**. It works directly on the **AST**
* Improving **pycode** is important in order to continue our investigation and since we rely on the conversion of the different iterables into **PythonRange**
* **PyccelAdd** and **PyccelMul** do not proceed to evaluation and are final nodes. This is why we get such ugly indices
* **PyccelFloorDiv** also caused me some troubles, and I had to add a prelude to the Outer loop in order to precompute the nominator and pass it as a variable to **PyccelFloorDiv**
* Splitting loops and the associated index replacements are done in one single pass, as well as gathering outer loops
* For the Unroll, I had to add an additional pass (the **finalize** method) can we remove it? one solution would be to add an **UnrolledFor** node and treat it in the printing stage, by replacing the target by integer values
* 3 temporary nodes have been added **InnerFor**, **OuterFor** and **SplittedFor**. They are only needed for the splitting pass and will not appear in the printing stage 
* I am using the Pyccel branch **devel-643** in which I improved a little bit the Python printer. But many things must be done
* Hidden imports are still missing in this branch (fixed in **issue #645**)

## Next steps 

* Integrate the class **Transform** into our pipeline
* Shall we use **decorators** or **header**?
* As any parser, the **Transform** class must be enriched with additional nodes, and this will only appear by extending our examples (function call, etc)
