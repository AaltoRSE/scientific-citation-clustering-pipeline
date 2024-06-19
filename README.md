# Scientific citation clustering pipeline

Group scientific article citations (references) using sentence-transformers embeddings + agglomerative clustering so that each cluster contains citations referring to the same article/book/report/etc.

Pipeline:

- For each citation, compute embedding vector using Huggingface sentence-transformers. Default model is `Snowflake/snowflake-arctic-embed-l`
- Cluster the embeddings using scikit-learn's agglomerative clustering implementation with cosine distance

Number of clusters is controlled using `distance_threshold` (between 0.0 and 1.0). Small values of `distance_threshold` (e.g. 0.05) result in high clustering precision (references in same cluster are likely to refer to same article). Higher values of distance_threshold (e.g. 0.2) result in high clustering recall (references to same article are very likely assigned to same cluster). Default value is `distance_threshold = 0.05`.

## Example

**Input citations**:

De Matos C.A., Rossi C.A.V., Word-of-mouth communications in marketing: A meta-analytic review of the antecedents and moderators, Journal of the Academy of Marketing Science, 36, 4, pp. 578-596, (2008)

Matos C.A.D., Rossi C.A.V., Word-of-mouth communications in marketing: a meta-analytic review of the antecedents and moderators, J. Acad. Market. Sci., 36, 4, pp. 578-596, (2008)

Eisingerich A.B., Bell S.J., Maintaining customer relationships in high credence services, Journal of Services Marketing, 21, 4, pp. 253-262, (2007)

Eisingerich A., Bell S., Maintaining Customer Relationships in High Credence Services, Journal of Service Marketing, 21, 4, pp. 253-262, (2007)

Eisingerich A., Bell S., Maintaining customer relationships in high credence services, Journal of Service Marketing, 21, 4, pp. 253-262, (2007)

Eisingerich A., Bell S., Maintaining customer relationships in high credence services, J. Serv. Market., 21, 4, pp. 253-262, (2007)

Hair J.F., Black W.C., Babin B.J., Anderson R.E., Multivariate Data Analysis, (2019)

**Resulting clusters**:

_Cluster 1:_

De Matos C.A., Rossi C.A.V., Word-of-mouth communications in marketing: A meta-analytic review of the antecedents and moderators, Journal of the Academy of Marketing Science, 36, 4, pp. 578-596, (2008)

Matos C.A.D., Rossi C.A.V., Word-of-mouth communications in marketing: a meta-analytic review of the antecedents and moderators, J. Acad. Market. Sci., 36, 4, pp. 578-596, (2008)

_Cluster 2:_

Eisingerich A.B., Bell S.J., Maintaining customer relationships in high credence services, Journal of Services Marketing, 21, 4, pp. 253-262, (2007)

Eisingerich A., Bell S., Maintaining Customer Relationships in High Credence Services, Journal of Service Marketing, 21, 4, pp. 253-262, (2007)

Eisingerich A., Bell S., Maintaining customer relationships in high credence services, Journal of Service Marketing, 21, 4, pp. 253-262, (2007)

Eisingerich A., Bell S., Maintaining customer relationships in high credence services, J. Serv. Market., 21, 4, pp. 253-262, (2007)

_Cluster 3:_

Hair J.F., Black W.C., Babin B.J., Anderson R.E., Multivariate Data Analysis, (2019)



## Setup

Setup on Aalto Triton.

### Using GPUs

Clone this repo and change directory with

```
cd $WRKDIR
git clone https://github.com/AaltoRSE/citation-grouping.git
cd citation-grouping
```

Create and activate conda environment **with GPU**
```
module load mamba
mamba env create -f env-gpu.yml -p env-gpu/
source activate env-gpu/ 
```

>**_NOTE:_** The GPU will be used to speed up embedding computation. The agglomerative clustering will always be run on CPUs.


### Using CPUs only

Clone this repo and change directory with

```
cd $WRKDIR
git clone https://github.com/AaltoRSE/citation-grouping.git
cd citation-grouping
```

Create and activate conda environment **with CPUs only**
```
module load mamba
mamba env create -f env-cpu.yml -p env-cpu/
source activate env-cpu/ 
```


## Run

### Using GPUs

Run **using GPUs**

```
cd $WRKDIR/citation-grouping
sbatch submit_gpu.sh /path/to/inputfile
```

where 

- `inputfile` is a CSV file with column "References"

The optional probability threshold `distance_threshold` (default is 0.05) can be tweaked inside the `submit_gpu.sh` file.

Results are written to (check the script output) 

```
/path/to/data/results/
```

including the final result file which is the original CSV file with added "References augmented" column and intermediate files.

### Using CPUs only

Run **using CPUs only**

```
module load mamba
module load model-huggingface
cd $WRKDIR/citation-grouping
sbatch submit_cpu.sh /path/to/inputfile
```

where `inputfile` is a CSV file with column "References".

The optional probability threshold `distance_threshold` (default is 0.05) can be tweaked inside the `submit_cpu.sh` file.

Results are written to (check the script output) 

```
/path/to/data/results/
```

including the final result file which is the original CSV file with added "References augmented" column and intermediate files.



## Resource usage

Agglomerative clustering uses O(N^2) memory but the implementation does a "preclustering" phase which reduces the memory requirements significantly if most of the references comprise a singleton cluster (which they do).

One could also use other, less memory intensive clustering methods but agglomerative works well and is easy to interpret.
