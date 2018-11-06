# TRG by itensor

This is a demo adapted from itensor's official tutorial. Difference mainly is extracting maximum value of T at each iteration.

## To complie

Change the first line of `Makefile` to change path to itensor. Then `make` .

## To use

You can call this demo with parameters, the first parameter will be `T` , the second be `iterations` or `log(lattice size)` and the third be `maximum cut`. e.g.

```bash
./trg 1.0 20 20
```

## Reference

[1]. [orginal itensor tutorial page](http://itensor.org/docs.cgi?page=book/trg)