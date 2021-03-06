??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02unknown8څ
?
conv_1_encoder/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv_1_encoder/kernel
?
)conv_1_encoder/kernel/Read/ReadVariableOpReadVariableOpconv_1_encoder/kernel*&
_output_shapes
:*
dtype0
~
conv_1_encoder/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv_1_encoder/bias
w
'conv_1_encoder/bias/Read/ReadVariableOpReadVariableOpconv_1_encoder/bias*
_output_shapes
:*
dtype0
?
conv_2_encoder/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv_2_encoder/kernel
?
)conv_2_encoder/kernel/Read/ReadVariableOpReadVariableOpconv_2_encoder/kernel*&
_output_shapes
:*
dtype0
~
conv_2_encoder/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv_2_encoder/bias
w
'conv_2_encoder/bias/Read/ReadVariableOpReadVariableOpconv_2_encoder/bias*
_output_shapes
:*
dtype0
?
conv_3_encoder/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameconv_3_encoder/kernel
?
)conv_3_encoder/kernel/Read/ReadVariableOpReadVariableOpconv_3_encoder/kernel*&
_output_shapes
: *
dtype0
~
conv_3_encoder/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameconv_3_encoder/bias
w
'conv_3_encoder/bias/Read/ReadVariableOpReadVariableOpconv_3_encoder/bias*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:  *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0
~
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/kernel
w
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*&
_output_shapes
:*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv_1_encoder/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv_1_encoder/kernel/m
?
0Adam/conv_1_encoder/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_1_encoder/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv_1_encoder/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/conv_1_encoder/bias/m
?
.Adam/conv_1_encoder/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_1_encoder/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv_2_encoder/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv_2_encoder/kernel/m
?
0Adam/conv_2_encoder/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_2_encoder/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv_2_encoder/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/conv_2_encoder/bias/m
?
.Adam/conv_2_encoder/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_2_encoder/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv_3_encoder/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/conv_3_encoder/kernel/m
?
0Adam/conv_3_encoder/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_3_encoder/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv_3_encoder/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/conv_3_encoder/bias/m
?
.Adam/conv_3_encoder/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_3_encoder/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:  *
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/m
?
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/output/kernel/m
?
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv_1_encoder/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv_1_encoder/kernel/v
?
0Adam/conv_1_encoder/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_1_encoder/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv_1_encoder/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/conv_1_encoder/bias/v
?
.Adam/conv_1_encoder/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_1_encoder/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv_2_encoder/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv_2_encoder/kernel/v
?
0Adam/conv_2_encoder/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_2_encoder/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv_2_encoder/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/conv_2_encoder/bias/v
?
.Adam/conv_2_encoder/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_2_encoder/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv_3_encoder/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/conv_3_encoder/kernel/v
?
0Adam/conv_3_encoder/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_3_encoder/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv_3_encoder/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/conv_3_encoder/bias/v
?
.Adam/conv_3_encoder/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_3_encoder/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:  *
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/v
?
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/output/kernel/v
?
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?W
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?W
value?WB?W B?W
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
R
%	variables
&trainable_variables
'regularization_losses
(	keras_api
h

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
R
/	variables
0trainable_variables
1regularization_losses
2	keras_api
h

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
R
9	variables
:trainable_variables
;regularization_losses
<	keras_api
h

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
R
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
h

Gkernel
Hbias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
R
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
h

Qkernel
Rbias
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
?
Witer

Xbeta_1

Ybeta_2
	Zdecay
[learning_ratem?m?m? m?)m?*m?3m?4m?=m?>m?Gm?Hm?Qm?Rm?v?v?v? v?)v?*v?3v?4v?=v?>v?Gv?Hv?Qv?Rv?
f
0
1
2
 3
)4
*5
36
47
=8
>9
G10
H11
Q12
R13
f
0
1
2
 3
)4
*5
36
47
=8
>9
G10
H11
Q12
R13
 
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
 
a_
VARIABLE_VALUEconv_1_encoder/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv_1_encoder/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
a_
VARIABLE_VALUEconv_2_encoder/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv_2_encoder/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
?
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
!	variables
"trainable_variables
#regularization_losses
 
 
 
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
%	variables
&trainable_variables
'regularization_losses
a_
VARIABLE_VALUEconv_3_encoder/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv_3_encoder/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
+	variables
,trainable_variables
-regularization_losses
 
 
 
?
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
/	variables
0trainable_variables
1regularization_losses
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
 
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
9	variables
:trainable_variables
;regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

=0
>1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

G0
H1

G0
H1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1

Q0
R1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUEAdam/conv_1_encoder/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv_1_encoder/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_2_encoder/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv_2_encoder/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_3_encoder/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv_3_encoder/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_1_encoder/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv_1_encoder/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_2_encoder/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv_2_encoder/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_3_encoder/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv_3_encoder/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv_1_encoder/kernelconv_1_encoder/biasconv_2_encoder/kernelconv_2_encoder/biasconv_3_encoder/kernelconv_3_encoder/biasconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasoutput/kerneloutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_52657
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)conv_1_encoder/kernel/Read/ReadVariableOp'conv_1_encoder/bias/Read/ReadVariableOp)conv_2_encoder/kernel/Read/ReadVariableOp'conv_2_encoder/bias/Read/ReadVariableOp)conv_3_encoder/kernel/Read/ReadVariableOp'conv_3_encoder/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp0Adam/conv_1_encoder/kernel/m/Read/ReadVariableOp.Adam/conv_1_encoder/bias/m/Read/ReadVariableOp0Adam/conv_2_encoder/kernel/m/Read/ReadVariableOp.Adam/conv_2_encoder/bias/m/Read/ReadVariableOp0Adam/conv_3_encoder/kernel/m/Read/ReadVariableOp.Adam/conv_3_encoder/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp0Adam/conv_1_encoder/kernel/v/Read/ReadVariableOp.Adam/conv_1_encoder/bias/v/Read/ReadVariableOp0Adam/conv_2_encoder/kernel/v/Read/ReadVariableOp.Adam/conv_2_encoder/bias/v/Read/ReadVariableOp0Adam/conv_3_encoder/kernel/v/Read/ReadVariableOp.Adam/conv_3_encoder/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*@
Tin9
725	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_53325
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_1_encoder/kernelconv_1_encoder/biasconv_2_encoder/kernelconv_2_encoder/biasconv_3_encoder/kernelconv_3_encoder/biasconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv_1_encoder/kernel/mAdam/conv_1_encoder/bias/mAdam/conv_2_encoder/kernel/mAdam/conv_2_encoder/bias/mAdam/conv_3_encoder/kernel/mAdam/conv_3_encoder/bias/mAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv_1_encoder/kernel/vAdam/conv_1_encoder/bias/vAdam/conv_2_encoder/kernel/vAdam/conv_2_encoder/bias/vAdam/conv_3_encoder/kernel/vAdam/conv_3_encoder/bias/vAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/output/kernel/vAdam/output/bias/v*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_53488??

?
a
E__inference_sampling_2_layer_call_and_return_conditional_losses_52057

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_52192

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
I__inference_conv_1_encoder_layer_call_and_return_conditional_losses_52879

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_conv2d_layer_call_and_return_conditional_losses_52999

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
E
)__inference_pooling_2_layer_call_fn_52929

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_2_layer_call_and_return_conditional_losses_52130h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_1_layer_call_fn_53038

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_52192w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
E
)__inference_pooling_2_layer_call_fn_52924

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_2_layer_call_and_return_conditional_losses_52007?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
.__inference_conv_1_encoder_layer_call_fn_52868

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_1_encoder_layer_call_and_return_conditional_losses_52097y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
E
)__inference_pooling_3_layer_call_fn_52969

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_3_layer_call_and_return_conditional_losses_52153h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@ :W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?U
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_52859

inputsG
-conv_1_encoder_conv2d_readvariableop_resource:<
.conv_1_encoder_biasadd_readvariableop_resource:G
-conv_2_encoder_conv2d_readvariableop_resource:<
.conv_2_encoder_biasadd_readvariableop_resource:G
-conv_3_encoder_conv2d_readvariableop_resource: <
.conv_3_encoder_biasadd_readvariableop_resource: ?
%conv2d_conv2d_readvariableop_resource:  4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:?
%output_conv2d_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?%conv_1_encoder/BiasAdd/ReadVariableOp?$conv_1_encoder/Conv2D/ReadVariableOp?%conv_2_encoder/BiasAdd/ReadVariableOp?$conv_2_encoder/Conv2D/ReadVariableOp?%conv_3_encoder/BiasAdd/ReadVariableOp?$conv_3_encoder/Conv2D/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/Conv2D/ReadVariableOp?
$conv_1_encoder/Conv2D/ReadVariableOpReadVariableOp-conv_1_encoder_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_1_encoder/Conv2DConv2Dinputs,conv_1_encoder/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
%conv_1_encoder/BiasAdd/ReadVariableOpReadVariableOp.conv_1_encoder_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_1_encoder/BiasAddBiasAddconv_1_encoder/Conv2D:output:0-conv_1_encoder/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????x
conv_1_encoder/ReluReluconv_1_encoder/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
pooling_1/MaxPoolMaxPool!conv_1_encoder/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
?
$conv_2_encoder/Conv2D/ReadVariableOpReadVariableOp-conv_2_encoder_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_2_encoder/Conv2DConv2Dpooling_1/MaxPool:output:0,conv_2_encoder/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
%conv_2_encoder/BiasAdd/ReadVariableOpReadVariableOp.conv_2_encoder_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_2_encoder/BiasAddBiasAddconv_2_encoder/Conv2D:output:0-conv_2_encoder/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????x
conv_2_encoder/ReluReluconv_2_encoder/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
pooling_2/MaxPoolMaxPool!conv_2_encoder/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingSAME*
strides
?
$conv_3_encoder/Conv2D/ReadVariableOpReadVariableOp-conv_3_encoder_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv_3_encoder/Conv2DConv2Dpooling_2/MaxPool:output:0,conv_3_encoder/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
%conv_3_encoder/BiasAdd/ReadVariableOpReadVariableOp.conv_3_encoder_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv_3_encoder/BiasAddBiasAddconv_3_encoder/Conv2D:output:0-conv_3_encoder/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ v
conv_3_encoder/ReluReluconv_3_encoder/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ ?
pooling_3/MaxPoolMaxPool!conv_3_encoder/Relu:activations:0*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d/Conv2DConv2Dpooling_3/MaxPool:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   a
sampling_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"        c
sampling_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      r
sampling_1/mulMulsampling_1/Const:output:0sampling_1/Const_1:output:0*
T0*
_output_shapes
:?
'sampling_1/resize/ResizeNearestNeighborResizeNearestNeighborconv2d/Relu:activations:0sampling_1/mul:z:0*
T0*/
_output_shapes
:?????????@@ *
half_pixel_centers(?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_1/Conv2DConv2D8sampling_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@a
sampling_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   c
sampling_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      r
sampling_2/mulMulsampling_2/Const:output:0sampling_2/Const_1:output:0*
T0*
_output_shapes
:?
'sampling_2/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_1/Relu:activations:0sampling_2/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_2/Conv2DConv2D8sampling_2/resize/ResizeNearestNeighbor:resized_images:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????l
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????a
sampling_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   c
sampling_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      r
sampling_3/mulMulsampling_3/Const:output:0sampling_3/Const_1:output:0*
T0*
_output_shapes
:?
'sampling_3/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_2/Relu:activations:0sampling_3/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
output/Conv2D/ReadVariableOpReadVariableOp%output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
output/Conv2DConv2D8sampling_3/resize/ResizeNearestNeighbor:resized_images:0$output/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output/BiasAddBiasAddoutput/Conv2D:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????n
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*1
_output_shapes
:???????????k
IdentityIdentityoutput/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp&^conv_1_encoder/BiasAdd/ReadVariableOp%^conv_1_encoder/Conv2D/ReadVariableOp&^conv_2_encoder/BiasAdd/ReadVariableOp%^conv_2_encoder/Conv2D/ReadVariableOp&^conv_3_encoder/BiasAdd/ReadVariableOp%^conv_3_encoder/Conv2D/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2N
%conv_1_encoder/BiasAdd/ReadVariableOp%conv_1_encoder/BiasAdd/ReadVariableOp2L
$conv_1_encoder/Conv2D/ReadVariableOp$conv_1_encoder/Conv2D/ReadVariableOp2N
%conv_2_encoder/BiasAdd/ReadVariableOp%conv_2_encoder/BiasAdd/ReadVariableOp2L
$conv_2_encoder/Conv2D/ReadVariableOp$conv_2_encoder/Conv2D/ReadVariableOp2N
%conv_3_encoder/BiasAdd/ReadVariableOp%conv_3_encoder/BiasAdd/ReadVariableOp2L
$conv_3_encoder/Conv2D/ReadVariableOp$conv_3_encoder/Conv2D/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/Conv2D/ReadVariableOpoutput/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
E
)__inference_pooling_3_layer_call_fn_52964

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_3_layer_call_and_return_conditional_losses_52019?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_pooling_2_layer_call_and_return_conditional_losses_52007

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
.__inference_conv_3_encoder_layer_call_fn_52948

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_3_encoder_layer_call_and_return_conditional_losses_52143w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
F
*__inference_sampling_2_layer_call_fn_53059

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_2_layer_call_and_return_conditional_losses_52205j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
(__inference_conv2d_2_layer_call_fn_53088

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_52218y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
D__inference_pooling_3_layer_call_and_return_conditional_losses_52974

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_autoencoder_layer_call_fn_52282	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_52251y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
`
D__inference_pooling_2_layer_call_and_return_conditional_losses_52130

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@@*
ksize
*
paddingSAME*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
a
E__inference_sampling_3_layer_call_and_return_conditional_losses_52231

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Q
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
*__inference_sampling_3_layer_call_fn_53109

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_3_layer_call_and_return_conditional_losses_52231j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?8
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_52571	
input.
conv_1_encoder_52529:"
conv_1_encoder_52531:.
conv_2_encoder_52535:"
conv_2_encoder_52537:.
conv_3_encoder_52541: "
conv_3_encoder_52543: &
conv2d_52547:  
conv2d_52549: (
conv2d_1_52553: 
conv2d_1_52555:(
conv2d_2_52559:
conv2d_2_52561:&
output_52565:
output_52567:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?&conv_1_encoder/StatefulPartitionedCall?&conv_2_encoder/StatefulPartitionedCall?&conv_3_encoder/StatefulPartitionedCall?output/StatefulPartitionedCall?
&conv_1_encoder/StatefulPartitionedCallStatefulPartitionedCallinputconv_1_encoder_52529conv_1_encoder_52531*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_1_encoder_layer_call_and_return_conditional_losses_52097?
pooling_1/PartitionedCallPartitionedCall/conv_1_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_1_layer_call_and_return_conditional_losses_52107?
&conv_2_encoder/StatefulPartitionedCallStatefulPartitionedCall"pooling_1/PartitionedCall:output:0conv_2_encoder_52535conv_2_encoder_52537*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_2_encoder_layer_call_and_return_conditional_losses_52120?
pooling_2/PartitionedCallPartitionedCall/conv_2_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_2_layer_call_and_return_conditional_losses_52130?
&conv_3_encoder/StatefulPartitionedCallStatefulPartitionedCall"pooling_2/PartitionedCall:output:0conv_3_encoder_52541conv_3_encoder_52543*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_3_encoder_layer_call_and_return_conditional_losses_52143?
pooling_3/PartitionedCallPartitionedCall/conv_3_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_3_layer_call_and_return_conditional_losses_52153?
conv2d/StatefulPartitionedCallStatefulPartitionedCall"pooling_3/PartitionedCall:output:0conv2d_52547conv2d_52549*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_52166?
sampling_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_1_layer_call_and_return_conditional_losses_52179?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#sampling_1/PartitionedCall:output:0conv2d_1_52553conv2d_1_52555*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_52192?
sampling_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_2_layer_call_and_return_conditional_losses_52205?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall#sampling_2/PartitionedCall:output:0conv2d_2_52559conv2d_2_52561*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_52218?
sampling_3/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_3_layer_call_and_return_conditional_losses_52231?
output/StatefulPartitionedCallStatefulPartitionedCall#sampling_3/PartitionedCall:output:0output_52565output_52567*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_52244?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall'^conv_1_encoder/StatefulPartitionedCall'^conv_2_encoder/StatefulPartitionedCall'^conv_3_encoder/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2P
&conv_1_encoder/StatefulPartitionedCall&conv_1_encoder/StatefulPartitionedCall2P
&conv_2_encoder/StatefulPartitionedCall&conv_2_encoder/StatefulPartitionedCall2P
&conv_3_encoder/StatefulPartitionedCall&conv_3_encoder/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_52218

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
+__inference_autoencoder_layer_call_fn_52526	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_52462y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
a
E__inference_sampling_2_layer_call_and_return_conditional_losses_53079

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Q
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_52657	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_51986y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
a
E__inference_sampling_2_layer_call_and_return_conditional_losses_53071

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_pooling_1_layer_call_and_return_conditional_losses_52899

inputs
identity?
MaxPoolMaxPoolinputs*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
b
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
D__inference_pooling_3_layer_call_and_return_conditional_losses_52153

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@ :W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
F
*__inference_sampling_3_layer_call_fn_53104

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_3_layer_call_and_return_conditional_losses_52076?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?8
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_52462

inputs.
conv_1_encoder_52420:"
conv_1_encoder_52422:.
conv_2_encoder_52426:"
conv_2_encoder_52428:.
conv_3_encoder_52432: "
conv_3_encoder_52434: &
conv2d_52438:  
conv2d_52440: (
conv2d_1_52444: 
conv2d_1_52446:(
conv2d_2_52450:
conv2d_2_52452:&
output_52456:
output_52458:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?&conv_1_encoder/StatefulPartitionedCall?&conv_2_encoder/StatefulPartitionedCall?&conv_3_encoder/StatefulPartitionedCall?output/StatefulPartitionedCall?
&conv_1_encoder/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_encoder_52420conv_1_encoder_52422*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_1_encoder_layer_call_and_return_conditional_losses_52097?
pooling_1/PartitionedCallPartitionedCall/conv_1_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_1_layer_call_and_return_conditional_losses_52107?
&conv_2_encoder/StatefulPartitionedCallStatefulPartitionedCall"pooling_1/PartitionedCall:output:0conv_2_encoder_52426conv_2_encoder_52428*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_2_encoder_layer_call_and_return_conditional_losses_52120?
pooling_2/PartitionedCallPartitionedCall/conv_2_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_2_layer_call_and_return_conditional_losses_52130?
&conv_3_encoder/StatefulPartitionedCallStatefulPartitionedCall"pooling_2/PartitionedCall:output:0conv_3_encoder_52432conv_3_encoder_52434*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_3_encoder_layer_call_and_return_conditional_losses_52143?
pooling_3/PartitionedCallPartitionedCall/conv_3_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_3_layer_call_and_return_conditional_losses_52153?
conv2d/StatefulPartitionedCallStatefulPartitionedCall"pooling_3/PartitionedCall:output:0conv2d_52438conv2d_52440*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_52166?
sampling_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_1_layer_call_and_return_conditional_losses_52179?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#sampling_1/PartitionedCall:output:0conv2d_1_52444conv2d_1_52446*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_52192?
sampling_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_2_layer_call_and_return_conditional_losses_52205?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall#sampling_2/PartitionedCall:output:0conv2d_2_52450conv2d_2_52452*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_52218?
sampling_3/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_3_layer_call_and_return_conditional_losses_52231?
output/StatefulPartitionedCallStatefulPartitionedCall#sampling_3/PartitionedCall:output:0output_52456output_52458*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_52244?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall'^conv_1_encoder/StatefulPartitionedCall'^conv_2_encoder/StatefulPartitionedCall'^conv_3_encoder/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2P
&conv_1_encoder/StatefulPartitionedCall&conv_1_encoder/StatefulPartitionedCall2P
&conv_2_encoder/StatefulPartitionedCall&conv_2_encoder/StatefulPartitionedCall2P
&conv_3_encoder/StatefulPartitionedCall&conv_3_encoder/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
a
E__inference_sampling_1_layer_call_and_return_conditional_losses_53021

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_sampling_1_layer_call_and_return_conditional_losses_52038

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?8
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_52251

inputs.
conv_1_encoder_52098:"
conv_1_encoder_52100:.
conv_2_encoder_52121:"
conv_2_encoder_52123:.
conv_3_encoder_52144: "
conv_3_encoder_52146: &
conv2d_52167:  
conv2d_52169: (
conv2d_1_52193: 
conv2d_1_52195:(
conv2d_2_52219:
conv2d_2_52221:&
output_52245:
output_52247:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?&conv_1_encoder/StatefulPartitionedCall?&conv_2_encoder/StatefulPartitionedCall?&conv_3_encoder/StatefulPartitionedCall?output/StatefulPartitionedCall?
&conv_1_encoder/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_encoder_52098conv_1_encoder_52100*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_1_encoder_layer_call_and_return_conditional_losses_52097?
pooling_1/PartitionedCallPartitionedCall/conv_1_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_1_layer_call_and_return_conditional_losses_52107?
&conv_2_encoder/StatefulPartitionedCallStatefulPartitionedCall"pooling_1/PartitionedCall:output:0conv_2_encoder_52121conv_2_encoder_52123*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_2_encoder_layer_call_and_return_conditional_losses_52120?
pooling_2/PartitionedCallPartitionedCall/conv_2_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_2_layer_call_and_return_conditional_losses_52130?
&conv_3_encoder/StatefulPartitionedCallStatefulPartitionedCall"pooling_2/PartitionedCall:output:0conv_3_encoder_52144conv_3_encoder_52146*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_3_encoder_layer_call_and_return_conditional_losses_52143?
pooling_3/PartitionedCallPartitionedCall/conv_3_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_3_layer_call_and_return_conditional_losses_52153?
conv2d/StatefulPartitionedCallStatefulPartitionedCall"pooling_3/PartitionedCall:output:0conv2d_52167conv2d_52169*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_52166?
sampling_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_1_layer_call_and_return_conditional_losses_52179?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#sampling_1/PartitionedCall:output:0conv2d_1_52193conv2d_1_52195*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_52192?
sampling_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_2_layer_call_and_return_conditional_losses_52205?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall#sampling_2/PartitionedCall:output:0conv2d_2_52219conv2d_2_52221*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_52218?
sampling_3/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_3_layer_call_and_return_conditional_losses_52231?
output/StatefulPartitionedCallStatefulPartitionedCall#sampling_3/PartitionedCall:output:0output_52245output_52247*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_52244?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall'^conv_1_encoder/StatefulPartitionedCall'^conv_2_encoder/StatefulPartitionedCall'^conv_3_encoder/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2P
&conv_1_encoder/StatefulPartitionedCall&conv_1_encoder/StatefulPartitionedCall2P
&conv_2_encoder/StatefulPartitionedCall&conv_2_encoder/StatefulPartitionedCall2P
&conv_3_encoder/StatefulPartitionedCall&conv_3_encoder/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
D__inference_pooling_2_layer_call_and_return_conditional_losses_52934

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
F
*__inference_sampling_1_layer_call_fn_53004

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_1_layer_call_and_return_conditional_losses_52038?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_pooling_3_layer_call_and_return_conditional_losses_52979

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@ :W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?h
?
__inference__traced_save_53325
file_prefix4
0savev2_conv_1_encoder_kernel_read_readvariableop2
.savev2_conv_1_encoder_bias_read_readvariableop4
0savev2_conv_2_encoder_kernel_read_readvariableop2
.savev2_conv_2_encoder_bias_read_readvariableop4
0savev2_conv_3_encoder_kernel_read_readvariableop2
.savev2_conv_3_encoder_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop;
7savev2_adam_conv_1_encoder_kernel_m_read_readvariableop9
5savev2_adam_conv_1_encoder_bias_m_read_readvariableop;
7savev2_adam_conv_2_encoder_kernel_m_read_readvariableop9
5savev2_adam_conv_2_encoder_bias_m_read_readvariableop;
7savev2_adam_conv_3_encoder_kernel_m_read_readvariableop9
5savev2_adam_conv_3_encoder_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop;
7savev2_adam_conv_1_encoder_kernel_v_read_readvariableop9
5savev2_adam_conv_1_encoder_bias_v_read_readvariableop;
7savev2_adam_conv_2_encoder_kernel_v_read_readvariableop9
5savev2_adam_conv_2_encoder_bias_v_read_readvariableop;
7savev2_adam_conv_3_encoder_kernel_v_read_readvariableop9
5savev2_adam_conv_3_encoder_bias_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*?
value?B?4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_conv_1_encoder_kernel_read_readvariableop.savev2_conv_1_encoder_bias_read_readvariableop0savev2_conv_2_encoder_kernel_read_readvariableop.savev2_conv_2_encoder_bias_read_readvariableop0savev2_conv_3_encoder_kernel_read_readvariableop.savev2_conv_3_encoder_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop7savev2_adam_conv_1_encoder_kernel_m_read_readvariableop5savev2_adam_conv_1_encoder_bias_m_read_readvariableop7savev2_adam_conv_2_encoder_kernel_m_read_readvariableop5savev2_adam_conv_2_encoder_bias_m_read_readvariableop7savev2_adam_conv_3_encoder_kernel_m_read_readvariableop5savev2_adam_conv_3_encoder_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop7savev2_adam_conv_1_encoder_kernel_v_read_readvariableop5savev2_adam_conv_1_encoder_bias_v_read_readvariableop7savev2_adam_conv_2_encoder_kernel_v_read_readvariableop5savev2_adam_conv_2_encoder_bias_v_read_readvariableop7savev2_adam_conv_3_encoder_kernel_v_read_readvariableop5savev2_adam_conv_3_encoder_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::: : :  : : :::::: : : : : : : : : ::::: : :  : : :::::::::: : :  : : :::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,	(
&
_output_shapes
: : 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :, (
&
_output_shapes
: : !

_output_shapes
::,"(
&
_output_shapes
:: #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
: : +

_output_shapes
: :,,(
&
_output_shapes
:  : -

_output_shapes
: :,.(
&
_output_shapes
: : /

_output_shapes
::,0(
&
_output_shapes
:: 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::4

_output_shapes
: 
?
?
I__inference_conv_1_encoder_layer_call_and_return_conditional_losses_52097

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
D__inference_pooling_1_layer_call_and_return_conditional_losses_52894

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
? 
!__inference__traced_restore_53488
file_prefix@
&assignvariableop_conv_1_encoder_kernel:4
&assignvariableop_1_conv_1_encoder_bias:B
(assignvariableop_2_conv_2_encoder_kernel:4
&assignvariableop_3_conv_2_encoder_bias:B
(assignvariableop_4_conv_3_encoder_kernel: 4
&assignvariableop_5_conv_3_encoder_bias: :
 assignvariableop_6_conv2d_kernel:  ,
assignvariableop_7_conv2d_bias: <
"assignvariableop_8_conv2d_1_kernel: .
 assignvariableop_9_conv2d_1_bias:=
#assignvariableop_10_conv2d_2_kernel:/
!assignvariableop_11_conv2d_2_bias:;
!assignvariableop_12_output_kernel:-
assignvariableop_13_output_bias:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: #
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: J
0assignvariableop_23_adam_conv_1_encoder_kernel_m:<
.assignvariableop_24_adam_conv_1_encoder_bias_m:J
0assignvariableop_25_adam_conv_2_encoder_kernel_m:<
.assignvariableop_26_adam_conv_2_encoder_bias_m:J
0assignvariableop_27_adam_conv_3_encoder_kernel_m: <
.assignvariableop_28_adam_conv_3_encoder_bias_m: B
(assignvariableop_29_adam_conv2d_kernel_m:  4
&assignvariableop_30_adam_conv2d_bias_m: D
*assignvariableop_31_adam_conv2d_1_kernel_m: 6
(assignvariableop_32_adam_conv2d_1_bias_m:D
*assignvariableop_33_adam_conv2d_2_kernel_m:6
(assignvariableop_34_adam_conv2d_2_bias_m:B
(assignvariableop_35_adam_output_kernel_m:4
&assignvariableop_36_adam_output_bias_m:J
0assignvariableop_37_adam_conv_1_encoder_kernel_v:<
.assignvariableop_38_adam_conv_1_encoder_bias_v:J
0assignvariableop_39_adam_conv_2_encoder_kernel_v:<
.assignvariableop_40_adam_conv_2_encoder_bias_v:J
0assignvariableop_41_adam_conv_3_encoder_kernel_v: <
.assignvariableop_42_adam_conv_3_encoder_bias_v: B
(assignvariableop_43_adam_conv2d_kernel_v:  4
&assignvariableop_44_adam_conv2d_bias_v: D
*assignvariableop_45_adam_conv2d_1_kernel_v: 6
(assignvariableop_46_adam_conv2d_1_bias_v:D
*assignvariableop_47_adam_conv2d_2_kernel_v:6
(assignvariableop_48_adam_conv2d_2_bias_v:B
(assignvariableop_49_adam_output_kernel_v:4
&assignvariableop_50_adam_output_bias_v:
identity_52??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*?
value?B?4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp&assignvariableop_conv_1_encoder_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp&assignvariableop_1_conv_1_encoder_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp(assignvariableop_2_conv_2_encoder_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp&assignvariableop_3_conv_2_encoder_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp(assignvariableop_4_conv_3_encoder_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp&assignvariableop_5_conv_3_encoder_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2d_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2d_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_output_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_output_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp0assignvariableop_23_adam_conv_1_encoder_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp.assignvariableop_24_adam_conv_1_encoder_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_conv_2_encoder_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp.assignvariableop_26_adam_conv_2_encoder_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp0assignvariableop_27_adam_conv_3_encoder_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp.assignvariableop_28_adam_conv_3_encoder_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv2d_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_conv2d_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_2_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_2_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_output_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_output_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp0assignvariableop_37_adam_conv_1_encoder_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp.assignvariableop_38_adam_conv_1_encoder_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp0assignvariableop_39_adam_conv_2_encoder_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp.assignvariableop_40_adam_conv_2_encoder_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp0assignvariableop_41_adam_conv_3_encoder_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp.assignvariableop_42_adam_conv_3_encoder_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_conv2d_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_conv2d_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv2d_1_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv2d_1_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv2d_2_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv2d_2_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_output_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_output_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_52IdentityIdentity_51:output:0^NoOp_1*
T0*
_output_shapes
: ?	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_52Identity_52:output:0*{
_input_shapesj
h: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
`
D__inference_pooling_1_layer_call_and_return_conditional_losses_51995

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_sampling_3_layer_call_and_return_conditional_losses_53121

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_sampling_2_layer_call_and_return_conditional_losses_52205

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Q
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
a
E__inference_sampling_1_layer_call_and_return_conditional_losses_52179

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"        X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Q
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*/
_output_shapes
:?????????@@ *
half_pixel_centers(}
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:?????????@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
F
*__inference_sampling_1_layer_call_fn_53009

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_1_layer_call_and_return_conditional_losses_52179h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
I__inference_conv_3_encoder_layer_call_and_return_conditional_losses_52143

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
`
D__inference_pooling_2_layer_call_and_return_conditional_losses_52939

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@@*
ksize
*
paddingSAME*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_conv2d_layer_call_and_return_conditional_losses_52166

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
+__inference_autoencoder_layer_call_fn_52690

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_52251y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
+__inference_autoencoder_layer_call_fn_52723

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_52462y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_output_layer_call_and_return_conditional_losses_52244

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:???????????d
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
E
)__inference_pooling_1_layer_call_fn_52889

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_1_layer_call_and_return_conditional_losses_52107j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
I__inference_conv_2_encoder_layer_call_and_return_conditional_losses_52919

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
.__inference_conv_2_encoder_layer_call_fn_52908

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_2_encoder_layer_call_and_return_conditional_losses_52120y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_53049

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?8
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_52616	
input.
conv_1_encoder_52574:"
conv_1_encoder_52576:.
conv_2_encoder_52580:"
conv_2_encoder_52582:.
conv_3_encoder_52586: "
conv_3_encoder_52588: &
conv2d_52592:  
conv2d_52594: (
conv2d_1_52598: 
conv2d_1_52600:(
conv2d_2_52604:
conv2d_2_52606:&
output_52610:
output_52612:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?&conv_1_encoder/StatefulPartitionedCall?&conv_2_encoder/StatefulPartitionedCall?&conv_3_encoder/StatefulPartitionedCall?output/StatefulPartitionedCall?
&conv_1_encoder/StatefulPartitionedCallStatefulPartitionedCallinputconv_1_encoder_52574conv_1_encoder_52576*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_1_encoder_layer_call_and_return_conditional_losses_52097?
pooling_1/PartitionedCallPartitionedCall/conv_1_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_1_layer_call_and_return_conditional_losses_52107?
&conv_2_encoder/StatefulPartitionedCallStatefulPartitionedCall"pooling_1/PartitionedCall:output:0conv_2_encoder_52580conv_2_encoder_52582*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_2_encoder_layer_call_and_return_conditional_losses_52120?
pooling_2/PartitionedCallPartitionedCall/conv_2_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_2_layer_call_and_return_conditional_losses_52130?
&conv_3_encoder/StatefulPartitionedCallStatefulPartitionedCall"pooling_2/PartitionedCall:output:0conv_3_encoder_52586conv_3_encoder_52588*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_3_encoder_layer_call_and_return_conditional_losses_52143?
pooling_3/PartitionedCallPartitionedCall/conv_3_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_3_layer_call_and_return_conditional_losses_52153?
conv2d/StatefulPartitionedCallStatefulPartitionedCall"pooling_3/PartitionedCall:output:0conv2d_52592conv2d_52594*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_52166?
sampling_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_1_layer_call_and_return_conditional_losses_52179?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#sampling_1/PartitionedCall:output:0conv2d_1_52598conv2d_1_52600*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_52192?
sampling_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_2_layer_call_and_return_conditional_losses_52205?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall#sampling_2/PartitionedCall:output:0conv2d_2_52604conv2d_2_52606*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_52218?
sampling_3/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_3_layer_call_and_return_conditional_losses_52231?
output/StatefulPartitionedCallStatefulPartitionedCall#sampling_3/PartitionedCall:output:0output_52610output_52612*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_52244?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall'^conv_1_encoder/StatefulPartitionedCall'^conv_2_encoder/StatefulPartitionedCall'^conv_3_encoder/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2P
&conv_1_encoder/StatefulPartitionedCall&conv_1_encoder/StatefulPartitionedCall2P
&conv_2_encoder/StatefulPartitionedCall&conv_2_encoder/StatefulPartitionedCall2P
&conv_3_encoder/StatefulPartitionedCall&conv_3_encoder/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
F
*__inference_sampling_2_layer_call_fn_53054

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sampling_2_layer_call_and_return_conditional_losses_52057?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_conv_3_encoder_layer_call_and_return_conditional_losses_52959

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
a
E__inference_sampling_3_layer_call_and_return_conditional_losses_52076

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_conv_2_encoder_layer_call_and_return_conditional_losses_52120

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_output_layer_call_fn_53138

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_52244y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
D__inference_pooling_3_layer_call_and_return_conditional_losses_52019

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?U
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_52791

inputsG
-conv_1_encoder_conv2d_readvariableop_resource:<
.conv_1_encoder_biasadd_readvariableop_resource:G
-conv_2_encoder_conv2d_readvariableop_resource:<
.conv_2_encoder_biasadd_readvariableop_resource:G
-conv_3_encoder_conv2d_readvariableop_resource: <
.conv_3_encoder_biasadd_readvariableop_resource: ?
%conv2d_conv2d_readvariableop_resource:  4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:?
%output_conv2d_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?%conv_1_encoder/BiasAdd/ReadVariableOp?$conv_1_encoder/Conv2D/ReadVariableOp?%conv_2_encoder/BiasAdd/ReadVariableOp?$conv_2_encoder/Conv2D/ReadVariableOp?%conv_3_encoder/BiasAdd/ReadVariableOp?$conv_3_encoder/Conv2D/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/Conv2D/ReadVariableOp?
$conv_1_encoder/Conv2D/ReadVariableOpReadVariableOp-conv_1_encoder_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_1_encoder/Conv2DConv2Dinputs,conv_1_encoder/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
%conv_1_encoder/BiasAdd/ReadVariableOpReadVariableOp.conv_1_encoder_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_1_encoder/BiasAddBiasAddconv_1_encoder/Conv2D:output:0-conv_1_encoder/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????x
conv_1_encoder/ReluReluconv_1_encoder/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
pooling_1/MaxPoolMaxPool!conv_1_encoder/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
?
$conv_2_encoder/Conv2D/ReadVariableOpReadVariableOp-conv_2_encoder_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_2_encoder/Conv2DConv2Dpooling_1/MaxPool:output:0,conv_2_encoder/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
%conv_2_encoder/BiasAdd/ReadVariableOpReadVariableOp.conv_2_encoder_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_2_encoder/BiasAddBiasAddconv_2_encoder/Conv2D:output:0-conv_2_encoder/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????x
conv_2_encoder/ReluReluconv_2_encoder/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
pooling_2/MaxPoolMaxPool!conv_2_encoder/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingSAME*
strides
?
$conv_3_encoder/Conv2D/ReadVariableOpReadVariableOp-conv_3_encoder_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv_3_encoder/Conv2DConv2Dpooling_2/MaxPool:output:0,conv_3_encoder/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
%conv_3_encoder/BiasAdd/ReadVariableOpReadVariableOp.conv_3_encoder_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv_3_encoder/BiasAddBiasAddconv_3_encoder/Conv2D:output:0-conv_3_encoder/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ v
conv_3_encoder/ReluReluconv_3_encoder/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ ?
pooling_3/MaxPoolMaxPool!conv_3_encoder/Relu:activations:0*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d/Conv2DConv2Dpooling_3/MaxPool:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   a
sampling_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"        c
sampling_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      r
sampling_1/mulMulsampling_1/Const:output:0sampling_1/Const_1:output:0*
T0*
_output_shapes
:?
'sampling_1/resize/ResizeNearestNeighborResizeNearestNeighborconv2d/Relu:activations:0sampling_1/mul:z:0*
T0*/
_output_shapes
:?????????@@ *
half_pixel_centers(?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_1/Conv2DConv2D8sampling_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@a
sampling_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   c
sampling_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      r
sampling_2/mulMulsampling_2/Const:output:0sampling_2/Const_1:output:0*
T0*
_output_shapes
:?
'sampling_2/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_1/Relu:activations:0sampling_2/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_2/Conv2DConv2D8sampling_2/resize/ResizeNearestNeighbor:resized_images:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????l
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????a
sampling_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   c
sampling_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      r
sampling_3/mulMulsampling_3/Const:output:0sampling_3/Const_1:output:0*
T0*
_output_shapes
:?
'sampling_3/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_2/Relu:activations:0sampling_3/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
output/Conv2D/ReadVariableOpReadVariableOp%output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
output/Conv2DConv2D8sampling_3/resize/ResizeNearestNeighbor:resized_images:0$output/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output/BiasAddBiasAddoutput/Conv2D:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????n
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*1
_output_shapes
:???????????k
IdentityIdentityoutput/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp&^conv_1_encoder/BiasAdd/ReadVariableOp%^conv_1_encoder/Conv2D/ReadVariableOp&^conv_2_encoder/BiasAdd/ReadVariableOp%^conv_2_encoder/Conv2D/ReadVariableOp&^conv_3_encoder/BiasAdd/ReadVariableOp%^conv_3_encoder/Conv2D/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2N
%conv_1_encoder/BiasAdd/ReadVariableOp%conv_1_encoder/BiasAdd/ReadVariableOp2L
$conv_1_encoder/Conv2D/ReadVariableOp$conv_1_encoder/Conv2D/ReadVariableOp2N
%conv_2_encoder/BiasAdd/ReadVariableOp%conv_2_encoder/BiasAdd/ReadVariableOp2L
$conv_2_encoder/Conv2D/ReadVariableOp$conv_2_encoder/Conv2D/ReadVariableOp2N
%conv_3_encoder/BiasAdd/ReadVariableOp%conv_3_encoder/BiasAdd/ReadVariableOp2L
$conv_3_encoder/Conv2D/ReadVariableOp$conv_3_encoder/Conv2D/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/Conv2D/ReadVariableOpoutput/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_output_layer_call_and_return_conditional_losses_53149

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:???????????d
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_53099

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?f
?
 __inference__wrapped_model_51986	
inputS
9autoencoder_conv_1_encoder_conv2d_readvariableop_resource:H
:autoencoder_conv_1_encoder_biasadd_readvariableop_resource:S
9autoencoder_conv_2_encoder_conv2d_readvariableop_resource:H
:autoencoder_conv_2_encoder_biasadd_readvariableop_resource:S
9autoencoder_conv_3_encoder_conv2d_readvariableop_resource: H
:autoencoder_conv_3_encoder_biasadd_readvariableop_resource: K
1autoencoder_conv2d_conv2d_readvariableop_resource:  @
2autoencoder_conv2d_biasadd_readvariableop_resource: M
3autoencoder_conv2d_1_conv2d_readvariableop_resource: B
4autoencoder_conv2d_1_biasadd_readvariableop_resource:M
3autoencoder_conv2d_2_conv2d_readvariableop_resource:B
4autoencoder_conv2d_2_biasadd_readvariableop_resource:K
1autoencoder_output_conv2d_readvariableop_resource:@
2autoencoder_output_biasadd_readvariableop_resource:
identity??)autoencoder/conv2d/BiasAdd/ReadVariableOp?(autoencoder/conv2d/Conv2D/ReadVariableOp?+autoencoder/conv2d_1/BiasAdd/ReadVariableOp?*autoencoder/conv2d_1/Conv2D/ReadVariableOp?+autoencoder/conv2d_2/BiasAdd/ReadVariableOp?*autoencoder/conv2d_2/Conv2D/ReadVariableOp?1autoencoder/conv_1_encoder/BiasAdd/ReadVariableOp?0autoencoder/conv_1_encoder/Conv2D/ReadVariableOp?1autoencoder/conv_2_encoder/BiasAdd/ReadVariableOp?0autoencoder/conv_2_encoder/Conv2D/ReadVariableOp?1autoencoder/conv_3_encoder/BiasAdd/ReadVariableOp?0autoencoder/conv_3_encoder/Conv2D/ReadVariableOp?)autoencoder/output/BiasAdd/ReadVariableOp?(autoencoder/output/Conv2D/ReadVariableOp?
0autoencoder/conv_1_encoder/Conv2D/ReadVariableOpReadVariableOp9autoencoder_conv_1_encoder_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
!autoencoder/conv_1_encoder/Conv2DConv2Dinput8autoencoder/conv_1_encoder/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
1autoencoder/conv_1_encoder/BiasAdd/ReadVariableOpReadVariableOp:autoencoder_conv_1_encoder_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
"autoencoder/conv_1_encoder/BiasAddBiasAdd*autoencoder/conv_1_encoder/Conv2D:output:09autoencoder/conv_1_encoder/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
autoencoder/conv_1_encoder/ReluRelu+autoencoder/conv_1_encoder/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
autoencoder/pooling_1/MaxPoolMaxPool-autoencoder/conv_1_encoder/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
?
0autoencoder/conv_2_encoder/Conv2D/ReadVariableOpReadVariableOp9autoencoder_conv_2_encoder_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
!autoencoder/conv_2_encoder/Conv2DConv2D&autoencoder/pooling_1/MaxPool:output:08autoencoder/conv_2_encoder/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
1autoencoder/conv_2_encoder/BiasAdd/ReadVariableOpReadVariableOp:autoencoder_conv_2_encoder_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
"autoencoder/conv_2_encoder/BiasAddBiasAdd*autoencoder/conv_2_encoder/Conv2D:output:09autoencoder/conv_2_encoder/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
autoencoder/conv_2_encoder/ReluRelu+autoencoder/conv_2_encoder/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
autoencoder/pooling_2/MaxPoolMaxPool-autoencoder/conv_2_encoder/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingSAME*
strides
?
0autoencoder/conv_3_encoder/Conv2D/ReadVariableOpReadVariableOp9autoencoder_conv_3_encoder_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
!autoencoder/conv_3_encoder/Conv2DConv2D&autoencoder/pooling_2/MaxPool:output:08autoencoder/conv_3_encoder/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
1autoencoder/conv_3_encoder/BiasAdd/ReadVariableOpReadVariableOp:autoencoder_conv_3_encoder_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
"autoencoder/conv_3_encoder/BiasAddBiasAdd*autoencoder/conv_3_encoder/Conv2D:output:09autoencoder/conv_3_encoder/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
autoencoder/conv_3_encoder/ReluRelu+autoencoder/conv_3_encoder/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ ?
autoencoder/pooling_3/MaxPoolMaxPool-autoencoder/conv_3_encoder/Relu:activations:0*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
?
(autoencoder/conv2d/Conv2D/ReadVariableOpReadVariableOp1autoencoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
autoencoder/conv2d/Conv2DConv2D&autoencoder/pooling_3/MaxPool:output:00autoencoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
?
)autoencoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp2autoencoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
autoencoder/conv2d/BiasAddBiasAdd"autoencoder/conv2d/Conv2D:output:01autoencoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   ~
autoencoder/conv2d/ReluRelu#autoencoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   m
autoencoder/sampling_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"        o
autoencoder/sampling_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ?
autoencoder/sampling_1/mulMul%autoencoder/sampling_1/Const:output:0'autoencoder/sampling_1/Const_1:output:0*
T0*
_output_shapes
:?
3autoencoder/sampling_1/resize/ResizeNearestNeighborResizeNearestNeighbor%autoencoder/conv2d/Relu:activations:0autoencoder/sampling_1/mul:z:0*
T0*/
_output_shapes
:?????????@@ *
half_pixel_centers(?
*autoencoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp3autoencoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
autoencoder/conv2d_1/Conv2DConv2DDautoencoder/sampling_1/resize/ResizeNearestNeighbor:resized_images:02autoencoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
+autoencoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp4autoencoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
autoencoder/conv2d_1/BiasAddBiasAdd$autoencoder/conv2d_1/Conv2D:output:03autoencoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@?
autoencoder/conv2d_1/ReluRelu%autoencoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@m
autoencoder/sampling_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   o
autoencoder/sampling_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ?
autoencoder/sampling_2/mulMul%autoencoder/sampling_2/Const:output:0'autoencoder/sampling_2/Const_1:output:0*
T0*
_output_shapes
:?
3autoencoder/sampling_2/resize/ResizeNearestNeighborResizeNearestNeighbor'autoencoder/conv2d_1/Relu:activations:0autoencoder/sampling_2/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
*autoencoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3autoencoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
autoencoder/conv2d_2/Conv2DConv2DDautoencoder/sampling_2/resize/ResizeNearestNeighbor:resized_images:02autoencoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
+autoencoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4autoencoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
autoencoder/conv2d_2/BiasAddBiasAdd$autoencoder/conv2d_2/Conv2D:output:03autoencoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
autoencoder/conv2d_2/ReluRelu%autoencoder/conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????m
autoencoder/sampling_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   o
autoencoder/sampling_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ?
autoencoder/sampling_3/mulMul%autoencoder/sampling_3/Const:output:0'autoencoder/sampling_3/Const_1:output:0*
T0*
_output_shapes
:?
3autoencoder/sampling_3/resize/ResizeNearestNeighborResizeNearestNeighbor'autoencoder/conv2d_2/Relu:activations:0autoencoder/sampling_3/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
(autoencoder/output/Conv2D/ReadVariableOpReadVariableOp1autoencoder_output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
autoencoder/output/Conv2DConv2DDautoencoder/sampling_3/resize/ResizeNearestNeighbor:resized_images:00autoencoder/output/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
)autoencoder/output/BiasAdd/ReadVariableOpReadVariableOp2autoencoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
autoencoder/output/BiasAddBiasAdd"autoencoder/output/Conv2D:output:01autoencoder/output/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
autoencoder/output/SigmoidSigmoid#autoencoder/output/BiasAdd:output:0*
T0*1
_output_shapes
:???????????w
IdentityIdentityautoencoder/output/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp*^autoencoder/conv2d/BiasAdd/ReadVariableOp)^autoencoder/conv2d/Conv2D/ReadVariableOp,^autoencoder/conv2d_1/BiasAdd/ReadVariableOp+^autoencoder/conv2d_1/Conv2D/ReadVariableOp,^autoencoder/conv2d_2/BiasAdd/ReadVariableOp+^autoencoder/conv2d_2/Conv2D/ReadVariableOp2^autoencoder/conv_1_encoder/BiasAdd/ReadVariableOp1^autoencoder/conv_1_encoder/Conv2D/ReadVariableOp2^autoencoder/conv_2_encoder/BiasAdd/ReadVariableOp1^autoencoder/conv_2_encoder/Conv2D/ReadVariableOp2^autoencoder/conv_3_encoder/BiasAdd/ReadVariableOp1^autoencoder/conv_3_encoder/Conv2D/ReadVariableOp*^autoencoder/output/BiasAdd/ReadVariableOp)^autoencoder/output/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 2V
)autoencoder/conv2d/BiasAdd/ReadVariableOp)autoencoder/conv2d/BiasAdd/ReadVariableOp2T
(autoencoder/conv2d/Conv2D/ReadVariableOp(autoencoder/conv2d/Conv2D/ReadVariableOp2Z
+autoencoder/conv2d_1/BiasAdd/ReadVariableOp+autoencoder/conv2d_1/BiasAdd/ReadVariableOp2X
*autoencoder/conv2d_1/Conv2D/ReadVariableOp*autoencoder/conv2d_1/Conv2D/ReadVariableOp2Z
+autoencoder/conv2d_2/BiasAdd/ReadVariableOp+autoencoder/conv2d_2/BiasAdd/ReadVariableOp2X
*autoencoder/conv2d_2/Conv2D/ReadVariableOp*autoencoder/conv2d_2/Conv2D/ReadVariableOp2f
1autoencoder/conv_1_encoder/BiasAdd/ReadVariableOp1autoencoder/conv_1_encoder/BiasAdd/ReadVariableOp2d
0autoencoder/conv_1_encoder/Conv2D/ReadVariableOp0autoencoder/conv_1_encoder/Conv2D/ReadVariableOp2f
1autoencoder/conv_2_encoder/BiasAdd/ReadVariableOp1autoencoder/conv_2_encoder/BiasAdd/ReadVariableOp2d
0autoencoder/conv_2_encoder/Conv2D/ReadVariableOp0autoencoder/conv_2_encoder/Conv2D/ReadVariableOp2f
1autoencoder/conv_3_encoder/BiasAdd/ReadVariableOp1autoencoder/conv_3_encoder/BiasAdd/ReadVariableOp2d
0autoencoder/conv_3_encoder/Conv2D/ReadVariableOp0autoencoder/conv_3_encoder/Conv2D/ReadVariableOp2V
)autoencoder/output/BiasAdd/ReadVariableOp)autoencoder/output/BiasAdd/ReadVariableOp2T
(autoencoder/output/Conv2D/ReadVariableOp(autoencoder/output/Conv2D/ReadVariableOp:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
?
&__inference_conv2d_layer_call_fn_52988

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_52166w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????   : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
a
E__inference_sampling_1_layer_call_and_return_conditional_losses_53029

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"        X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Q
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*/
_output_shapes
:?????????@@ *
half_pixel_centers(}
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:?????????@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
E
)__inference_pooling_1_layer_call_fn_52884

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_pooling_1_layer_call_and_return_conditional_losses_51995?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_sampling_3_layer_call_and_return_conditional_losses_53129

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Q
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
D__inference_pooling_1_layer_call_and_return_conditional_losses_52107

inputs
identity?
MaxPoolMaxPoolinputs*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
b
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A
input8
serving_default_input:0???????????D
output:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
9	variables
:trainable_variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Gkernel
Hbias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Qkernel
Rbias
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Witer

Xbeta_1

Ybeta_2
	Zdecay
[learning_ratem?m?m? m?)m?*m?3m?4m?=m?>m?Gm?Hm?Qm?Rm?v?v?v? v?)v?*v?3v?4v?=v?>v?Gv?Hv?Qv?Rv?"
	optimizer
?
0
1
2
 3
)4
*5
36
47
=8
>9
G10
H11
Q12
R13"
trackable_list_wrapper
?
0
1
2
 3
)4
*5
36
47
=8
>9
G10
H11
Q12
R13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
/:-2conv_1_encoder/kernel
!:2conv_1_encoder/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-2conv_2_encoder/kernel
!:2conv_2_encoder/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
!	variables
"trainable_variables
#regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
%	variables
&trainable_variables
'regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:- 2conv_3_encoder/kernel
!: 2conv_3_encoder/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
+	variables
,trainable_variables
-regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
/	variables
0trainable_variables
1regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%  2conv2d/kernel
: 2conv2d/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
9	variables
:trainable_variables
;regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_1/kernel
:2conv2d_1/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_2/kernel
:2conv2d_2/bias
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%2output/kernel
:2output/bias
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
4:22Adam/conv_1_encoder/kernel/m
&:$2Adam/conv_1_encoder/bias/m
4:22Adam/conv_2_encoder/kernel/m
&:$2Adam/conv_2_encoder/bias/m
4:2 2Adam/conv_3_encoder/kernel/m
&:$ 2Adam/conv_3_encoder/bias/m
,:*  2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
.:, 2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
.:,2Adam/conv2d_2/kernel/m
 :2Adam/conv2d_2/bias/m
,:*2Adam/output/kernel/m
:2Adam/output/bias/m
4:22Adam/conv_1_encoder/kernel/v
&:$2Adam/conv_1_encoder/bias/v
4:22Adam/conv_2_encoder/kernel/v
&:$2Adam/conv_2_encoder/bias/v
4:2 2Adam/conv_3_encoder/kernel/v
&:$ 2Adam/conv_3_encoder/bias/v
,:*  2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
.:, 2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
.:,2Adam/conv2d_2/kernel/v
 :2Adam/conv2d_2/bias/v
,:*2Adam/output/kernel/v
:2Adam/output/bias/v
?2?
+__inference_autoencoder_layer_call_fn_52282
+__inference_autoencoder_layer_call_fn_52690
+__inference_autoencoder_layer_call_fn_52723
+__inference_autoencoder_layer_call_fn_52526?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_autoencoder_layer_call_and_return_conditional_losses_52791
F__inference_autoencoder_layer_call_and_return_conditional_losses_52859
F__inference_autoencoder_layer_call_and_return_conditional_losses_52571
F__inference_autoencoder_layer_call_and_return_conditional_losses_52616?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_51986input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_conv_1_encoder_layer_call_fn_52868?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_conv_1_encoder_layer_call_and_return_conditional_losses_52879?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_pooling_1_layer_call_fn_52884
)__inference_pooling_1_layer_call_fn_52889?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_pooling_1_layer_call_and_return_conditional_losses_52894
D__inference_pooling_1_layer_call_and_return_conditional_losses_52899?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_conv_2_encoder_layer_call_fn_52908?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_conv_2_encoder_layer_call_and_return_conditional_losses_52919?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_pooling_2_layer_call_fn_52924
)__inference_pooling_2_layer_call_fn_52929?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_pooling_2_layer_call_and_return_conditional_losses_52934
D__inference_pooling_2_layer_call_and_return_conditional_losses_52939?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_conv_3_encoder_layer_call_fn_52948?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_conv_3_encoder_layer_call_and_return_conditional_losses_52959?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_pooling_3_layer_call_fn_52964
)__inference_pooling_3_layer_call_fn_52969?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_pooling_3_layer_call_and_return_conditional_losses_52974
D__inference_pooling_3_layer_call_and_return_conditional_losses_52979?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv2d_layer_call_fn_52988?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv2d_layer_call_and_return_conditional_losses_52999?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_sampling_1_layer_call_fn_53004
*__inference_sampling_1_layer_call_fn_53009?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_sampling_1_layer_call_and_return_conditional_losses_53021
E__inference_sampling_1_layer_call_and_return_conditional_losses_53029?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_1_layer_call_fn_53038?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_53049?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_sampling_2_layer_call_fn_53054
*__inference_sampling_2_layer_call_fn_53059?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_sampling_2_layer_call_and_return_conditional_losses_53071
E__inference_sampling_2_layer_call_and_return_conditional_losses_53079?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_2_layer_call_fn_53088?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_53099?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_sampling_3_layer_call_fn_53104
*__inference_sampling_3_layer_call_fn_53109?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_sampling_3_layer_call_and_return_conditional_losses_53121
E__inference_sampling_3_layer_call_and_return_conditional_losses_53129?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_output_layer_call_fn_53138?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_output_layer_call_and_return_conditional_losses_53149?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_52657input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_51986? )*34=>GHQR8?5
.?+
)?&
input???????????
? "9?6
4
output*?'
output????????????
F__inference_autoencoder_layer_call_and_return_conditional_losses_52571? )*34=>GHQR@?=
6?3
)?&
input???????????
p 

 
? "/?,
%?"
0???????????
? ?
F__inference_autoencoder_layer_call_and_return_conditional_losses_52616? )*34=>GHQR@?=
6?3
)?&
input???????????
p

 
? "/?,
%?"
0???????????
? ?
F__inference_autoencoder_layer_call_and_return_conditional_losses_52791? )*34=>GHQRA?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
F__inference_autoencoder_layer_call_and_return_conditional_losses_52859? )*34=>GHQRA?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
+__inference_autoencoder_layer_call_fn_52282v )*34=>GHQR@?=
6?3
)?&
input???????????
p 

 
? ""?????????????
+__inference_autoencoder_layer_call_fn_52526v )*34=>GHQR@?=
6?3
)?&
input???????????
p

 
? ""?????????????
+__inference_autoencoder_layer_call_fn_52690w )*34=>GHQRA?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
+__inference_autoencoder_layer_call_fn_52723w )*34=>GHQRA?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
C__inference_conv2d_1_layer_call_and_return_conditional_losses_53049l=>7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????@@
? ?
(__inference_conv2d_1_layer_call_fn_53038_=>7?4
-?*
(?%
inputs?????????@@ 
? " ??????????@@?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_53099pGH9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
(__inference_conv2d_2_layer_call_fn_53088cGH9?6
/?,
*?'
inputs???????????
? ""?????????????
A__inference_conv2d_layer_call_and_return_conditional_losses_52999l347?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
&__inference_conv2d_layer_call_fn_52988_347?4
-?*
(?%
inputs?????????   
? " ??????????   ?
I__inference_conv_1_encoder_layer_call_and_return_conditional_losses_52879p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
.__inference_conv_1_encoder_layer_call_fn_52868c9?6
/?,
*?'
inputs???????????
? ""?????????????
I__inference_conv_2_encoder_layer_call_and_return_conditional_losses_52919p 9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
.__inference_conv_2_encoder_layer_call_fn_52908c 9?6
/?,
*?'
inputs???????????
? ""?????????????
I__inference_conv_3_encoder_layer_call_and_return_conditional_losses_52959l)*7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@ 
? ?
.__inference_conv_3_encoder_layer_call_fn_52948_)*7?4
-?*
(?%
inputs?????????@@
? " ??????????@@ ?
A__inference_output_layer_call_and_return_conditional_losses_53149pQR9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
&__inference_output_layer_call_fn_53138cQR9?6
/?,
*?'
inputs???????????
? ""?????????????
D__inference_pooling_1_layer_call_and_return_conditional_losses_52894?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
D__inference_pooling_1_layer_call_and_return_conditional_losses_52899l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
)__inference_pooling_1_layer_call_fn_52884?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
)__inference_pooling_1_layer_call_fn_52889_9?6
/?,
*?'
inputs???????????
? ""?????????????
D__inference_pooling_2_layer_call_and_return_conditional_losses_52934?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
D__inference_pooling_2_layer_call_and_return_conditional_losses_52939j9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????@@
? ?
)__inference_pooling_2_layer_call_fn_52924?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
)__inference_pooling_2_layer_call_fn_52929]9?6
/?,
*?'
inputs???????????
? " ??????????@@?
D__inference_pooling_3_layer_call_and_return_conditional_losses_52974?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
D__inference_pooling_3_layer_call_and_return_conditional_losses_52979h7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????   
? ?
)__inference_pooling_3_layer_call_fn_52964?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
)__inference_pooling_3_layer_call_fn_52969[7?4
-?*
(?%
inputs?????????@@ 
? " ??????????   ?
E__inference_sampling_1_layer_call_and_return_conditional_losses_53021?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
E__inference_sampling_1_layer_call_and_return_conditional_losses_53029h7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????@@ 
? ?
*__inference_sampling_1_layer_call_fn_53004?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
*__inference_sampling_1_layer_call_fn_53009[7?4
-?*
(?%
inputs?????????   
? " ??????????@@ ?
E__inference_sampling_2_layer_call_and_return_conditional_losses_53071?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
E__inference_sampling_2_layer_call_and_return_conditional_losses_53079j7?4
-?*
(?%
inputs?????????@@
? "/?,
%?"
0???????????
? ?
*__inference_sampling_2_layer_call_fn_53054?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
*__inference_sampling_2_layer_call_fn_53059]7?4
-?*
(?%
inputs?????????@@
? ""?????????????
E__inference_sampling_3_layer_call_and_return_conditional_losses_53121?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
E__inference_sampling_3_layer_call_and_return_conditional_losses_53129l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
*__inference_sampling_3_layer_call_fn_53104?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
*__inference_sampling_3_layer_call_fn_53109_9?6
/?,
*?'
inputs???????????
? ""?????????????
#__inference_signature_wrapper_52657? )*34=>GHQRA?>
? 
7?4
2
input)?&
input???????????"9?6
4
output*?'
output???????????