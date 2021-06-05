# Supplementary information

The repository contains supplementary information for the manuscript : "Quantitative examination of five stochastic cell-cycle and cell-size control models for *Escherichia coli* and *Bacillus subtilis*".

Please start at the notebook [frontiers_index](Response/frontiers_index.ipynb) to be guided through the results.

To be able to execute the notebooks, specific libraries are required. For convenience, a jupyter server can be started locally using docker.
Execute in a terminal (at the root of the repository):
```
docker-compose up
```

Copy the prompted token and paste it in the corresponding field when entering the jupyter server. The jupyter server can be accessed by entering `http://localhost:9999` in a web browser.

You'll need a working installation of docker and docker-compose (see the [doc](https://docs.docker.com/compose/gettingstarted/)). Before being able to start a jupyter server through docker, you will need to create an image.
This can be done by executing (at the root of the repository) `docker-compose build`. The image doesn't need to be rebuilt after stopping a jupyter server.
