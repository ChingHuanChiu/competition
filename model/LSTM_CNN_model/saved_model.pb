??'
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??!
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@ *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
|
dense_1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense_1_1/kernel
u
$dense_1_1/kernel/Read/ReadVariableOpReadVariableOpdense_1_1/kernel*
_output_shapes

: *
dtype0
t
dense_1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1_1/bias
m
"dense_1_1/bias/Read/ReadVariableOpReadVariableOpdense_1_1/bias*
_output_shapes
:*
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
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:@*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:@*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:@@*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:@*
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:@@*
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:@*
dtype0
?
lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*&
shared_namelstm/lstm_cell/kernel
?
)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel*
_output_shapes
:	@?*
dtype0
?
lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*0
shared_name!lstm/lstm_cell/recurrent_kernel
?
3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel*
_output_shapes
:	 ?*
dtype0

lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namelstm/lstm_cell/bias
x
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_output_shapes	
:?*
dtype0
?
attention_score_vec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *+
shared_nameattention_score_vec/kernel
?
.attention_score_vec/kernel/Read/ReadVariableOpReadVariableOpattention_score_vec/kernel*
_output_shapes

:  *
dtype0
?
attention_vector/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*(
shared_nameattention_vector/kernel
?
+attention_vector/kernel/Read/ReadVariableOpReadVariableOpattention_vector/kernel*
_output_shapes
:	@?*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	?@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
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
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:@ *
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_1_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/dense_1_1/kernel/m
?
+Adam/dense_1_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1_1/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_1_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_1_1/bias/m
{
)Adam/dense_1_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:@ *
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_1_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/dense_1_1/kernel/v
?
+Adam/dense_1_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1_1/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_1_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_1_1/bias/v
{
)Adam/dense_1_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?Y
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?Y
value?YB?Y B?Y
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
layer-8
layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
R
%regularization_losses
&	variables
'trainable_variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
?
/iter

0beta_1

1beta_2
	2decay
3learning_ratem? m?)m?*m?v? v?)v?*v?
 
?
40
51
62
73
84
95
:6
;7
<8
=9
>10
?11
@12
A13
B14
15
 16
)17
*18

0
 1
)2
*3
?

Clayers
Dnon_trainable_variables
Elayer_metrics
Flayer_regularization_losses
regularization_losses
	variables
Gmetrics
trainable_variables
 
 
h

4kernel
5bias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
h

6kernel
7bias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
h

8kernel
9bias
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
R
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
l
Xcell
Y
state_spec
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
^

=kernel
^regularization_losses
_	variables
`trainable_variables
a	keras_api
R
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
\
faxes
gregularization_losses
h	variables
itrainable_variables
j	keras_api
R
kregularization_losses
l	variables
mtrainable_variables
n	keras_api
\
oaxes
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
R
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
^

>kernel
xregularization_losses
y	variables
ztrainable_variables
{	keras_api
R
|regularization_losses
}	variables
~trainable_variables
	keras_api
l

?kernel
@bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
l

Akernel
Bbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 
n
40
51
62
73
84
95
:6
;7
<8
=9
>10
?11
@12
A13
B14
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
regularization_losses
	variables
?metrics
trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
!regularization_losses
"	variables
?metrics
#trainable_variables
 
 
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
%regularization_losses
&	variables
?metrics
'trainable_variables
\Z
VARIABLE_VALUEdense_1_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_1_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
+regularization_losses
,	variables
?metrics
-trainable_variables
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
IG
VARIABLE_VALUEconv1d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv1d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv1d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv1d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv1d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv1d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUElstm/lstm_cell/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElstm/lstm_cell/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUElstm/lstm_cell/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEattention_score_vec/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEattention_vector/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_1/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_1/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_2/kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_2/bias'variables/14/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
n
40
51
62
73
84
95
:6
;7
<8
=9
>10
?11
@12
A13
B14
 
 

?0
?1
 

40
51
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
Hregularization_losses
I	variables
?metrics
Jtrainable_variables
 

60
71
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
Lregularization_losses
M	variables
?metrics
Ntrainable_variables
 

80
91
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
Pregularization_losses
Q	variables
?metrics
Rtrainable_variables
 
 
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
Tregularization_losses
U	variables
?metrics
Vtrainable_variables
?

:kernel
;recurrent_kernel
<bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 
 

:0
;1
<2
 
?
?layers
?states
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
Zregularization_losses
[	variables
?metrics
\trainable_variables
 

=0
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
^regularization_losses
_	variables
?metrics
`trainable_variables
 
 
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
bregularization_losses
c	variables
?metrics
dtrainable_variables
 
 
 
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
gregularization_losses
h	variables
?metrics
itrainable_variables
 
 
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
kregularization_losses
l	variables
?metrics
mtrainable_variables
 
 
 
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
pregularization_losses
q	variables
?metrics
rtrainable_variables
 
 
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
tregularization_losses
u	variables
?metrics
vtrainable_variables
 

>0
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
xregularization_losses
y	variables
?metrics
ztrainable_variables
 
 
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
|regularization_losses
}	variables
?metrics
~trainable_variables
 

?0
@1
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?metrics
?trainable_variables
 

A0
B1
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?metrics
?trainable_variables
v
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
n
40
51
62
73
84
95
:6
;7
<8
=9
>10
?11
@12
A13
B14
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
 

40
51
 
 
 
 

60
71
 
 
 
 

80
91
 
 
 
 
 
 
 
 
 

:0
;1
<2
 
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?metrics
?trainable_variables

X0
 

:0
;1
<2
 
 
 
 

=0
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

>0
 
 
 
 
 
 
 
 
 

?0
@1
 
 
 
 

A0
B1
 
 
 
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
 

:0
;1
<2
 
 
 
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_1_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_1_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_1_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_1_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_model_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_model_inputconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biaslstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biasattention_score_vec/kernelattention_vector/kerneldense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense/kernel
dense/biasdense_1_1/kerneldense_1_1/bias*
Tin
2*
Tout
2*'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference_signature_wrapper_17516
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp$dense_1_1/kernel/Read/ReadVariableOp"dense_1_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp)lstm/lstm_cell/kernel/Read/ReadVariableOp3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp'lstm/lstm_cell/bias/Read/ReadVariableOp.attention_score_vec/kernel/Read/ReadVariableOp+attention_vector/kernel/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp+Adam/dense_1_1/kernel/m/Read/ReadVariableOp)Adam/dense_1_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp+Adam/dense_1_1/kernel/v/Read/ReadVariableOp)Adam/dense_1_1/bias/v/Read/ReadVariableOpConst*1
Tin*
(2&	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__traced_save_19920
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1_1/kerneldense_1_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biaslstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biasattention_score_vec/kernelattention_vector/kerneldense_1/kerneldense_1/biasdense_2/kerneldense_2/biastotalcounttotal_1count_1Adam/dense/kernel/mAdam/dense/bias/mAdam/dense_1_1/kernel/mAdam/dense_1_1/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1_1/kernel/vAdam/dense_1_1/bias/v*0
Tin)
'2%*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_restore_20040?? 
?
?
model_lstm_while_cond_17629!
model_lstm_while_loop_counter'
#model_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3#
less_model_lstm_strided_slice_18
4model_lstm_while_cond_17629___redundant_placeholder08
4model_lstm_while_cond_17629___redundant_placeholder18
4model_lstm_while_cond_17629___redundant_placeholder28
4model_lstm_while_cond_17629___redundant_placeholder3
identity
c
LessLessplaceholderless_model_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_18821

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :::O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_17221

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :::O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
@__inference_model_layer_call_and_return_conditional_losses_18433

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource1
-lstm_lstm_cell_matmul_readvariableop_resource3
/lstm_lstm_cell_matmul_1_readvariableop_resource2
.lstm_lstm_cell_biasadd_readvariableop_resource9
5attention_score_vec_tensordot_readvariableop_resource3
/attention_vector_matmul_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??
lstm/while?
conv1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d/Pad/paddings{

conv1d/PadPadinputsconv1d/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2

conv1d/Pad~
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimsconv1d/Pad:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d/Relu?
conv1d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_1/Pad/paddings?
conv1d_1/PadPadconv1d/Relu:activations:0conv1d_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_1/Pad?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d_1/Pad:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_1/Relu?
conv1d_2/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_2/Pad/paddings?
conv1d_2/PadPadconv1d_1/Relu:activations:0conv1d_2/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_2/Pad?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimsconv1d_2/Pad:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_2/Relu~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dim?
max_pooling1d/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0%max_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
max_pooling1d/ExpandDims?
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPool?
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
max_pooling1d/Squeezef

lstm/ShapeShapemax_pooling1d/Squeeze:output:0*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack?
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1?
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2?
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros/mul/y?
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros/packed/1?
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const?

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros_1/mul/y?
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/zeros_1/Less/y?
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros_1/packed/1?
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const?
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm?
lstm/transpose	Transposemax_pooling1d/Squeeze:output:0lstm/transpose/perm:output:0*
T0*+
_output_shapes
:?????????@2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1?
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack?
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1?
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2?
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1?
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm/TensorArrayV2/element_shape?
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2?
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape?
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor?
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack?
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1?
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2?
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
lstm/strided_slice_2?
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOp?
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_cell/MatMul?
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOp?
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_cell/MatMul_1?
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_cell/add?
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOp?
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_cell/BiasAddn
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell/Const?
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim?
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm/lstm_cell/split?
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/Sigmoid?
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/Sigmoid_1?
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/mul?
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/Relu?
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/mul_1?
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/add_1?
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/Sigmoid_2?
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/Relu_1?
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/mul_2?
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2$
"lstm/TensorArrayV2_1/element_shape?
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time?
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counter?

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
lstm_while_body_18280*!
condR
lstm_while_cond_18279*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2

lstm/while?
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    27
5lstm/TensorArrayV2Stack/TensorListStack/element_shape?
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack?
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm/strided_slice_3/stack?
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1?
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2?
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
lstm/strided_slice_3?
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm?
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtime?
%last_hidden_state/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2'
%last_hidden_state/strided_slice/stack?
'last_hidden_state/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2)
'last_hidden_state/strided_slice/stack_1?
'last_hidden_state/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2)
'last_hidden_state/strided_slice/stack_2?
last_hidden_state/strided_sliceStridedSlicelstm/transpose_1:y:0.last_hidden_state/strided_slice/stack:output:00last_hidden_state/strided_slice/stack_1:output:00last_hidden_state/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *

begin_mask*
end_mask*
shrink_axis_mask2!
last_hidden_state/strided_slice?
,attention_score_vec/Tensordot/ReadVariableOpReadVariableOp5attention_score_vec_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02.
,attention_score_vec/Tensordot/ReadVariableOp?
"attention_score_vec/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"attention_score_vec/Tensordot/axes?
"attention_score_vec/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"attention_score_vec/Tensordot/free?
#attention_score_vec/Tensordot/ShapeShapelstm/transpose_1:y:0*
T0*
_output_shapes
:2%
#attention_score_vec/Tensordot/Shape?
+attention_score_vec/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+attention_score_vec/Tensordot/GatherV2/axis?
&attention_score_vec/Tensordot/GatherV2GatherV2,attention_score_vec/Tensordot/Shape:output:0+attention_score_vec/Tensordot/free:output:04attention_score_vec/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&attention_score_vec/Tensordot/GatherV2?
-attention_score_vec/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-attention_score_vec/Tensordot/GatherV2_1/axis?
(attention_score_vec/Tensordot/GatherV2_1GatherV2,attention_score_vec/Tensordot/Shape:output:0+attention_score_vec/Tensordot/axes:output:06attention_score_vec/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(attention_score_vec/Tensordot/GatherV2_1?
#attention_score_vec/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#attention_score_vec/Tensordot/Const?
"attention_score_vec/Tensordot/ProdProd/attention_score_vec/Tensordot/GatherV2:output:0,attention_score_vec/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"attention_score_vec/Tensordot/Prod?
%attention_score_vec/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%attention_score_vec/Tensordot/Const_1?
$attention_score_vec/Tensordot/Prod_1Prod1attention_score_vec/Tensordot/GatherV2_1:output:0.attention_score_vec/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$attention_score_vec/Tensordot/Prod_1?
)attention_score_vec/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)attention_score_vec/Tensordot/concat/axis?
$attention_score_vec/Tensordot/concatConcatV2+attention_score_vec/Tensordot/free:output:0+attention_score_vec/Tensordot/axes:output:02attention_score_vec/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$attention_score_vec/Tensordot/concat?
#attention_score_vec/Tensordot/stackPack+attention_score_vec/Tensordot/Prod:output:0-attention_score_vec/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#attention_score_vec/Tensordot/stack?
'attention_score_vec/Tensordot/transpose	Transposelstm/transpose_1:y:0-attention_score_vec/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2)
'attention_score_vec/Tensordot/transpose?
%attention_score_vec/Tensordot/ReshapeReshape+attention_score_vec/Tensordot/transpose:y:0,attention_score_vec/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%attention_score_vec/Tensordot/Reshape?
$attention_score_vec/Tensordot/MatMulMatMul.attention_score_vec/Tensordot/Reshape:output:04attention_score_vec/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2&
$attention_score_vec/Tensordot/MatMul?
%attention_score_vec/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2'
%attention_score_vec/Tensordot/Const_2?
+attention_score_vec/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+attention_score_vec/Tensordot/concat_1/axis?
&attention_score_vec/Tensordot/concat_1ConcatV2/attention_score_vec/Tensordot/GatherV2:output:0.attention_score_vec/Tensordot/Const_2:output:04attention_score_vec/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&attention_score_vec/Tensordot/concat_1?
attention_score_vec/TensordotReshape.attention_score_vec/Tensordot/MatMul:product:0/attention_score_vec/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
attention_score_vec/Tensordot?
attention_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
attention_score/ExpandDims/dim?
attention_score/ExpandDims
ExpandDims(last_hidden_state/strided_slice:output:0'attention_score/ExpandDims/dim:output:0*
T0*+
_output_shapes
:????????? 2
attention_score/ExpandDims?
attention_score/MatMulBatchMatMulV2&attention_score_vec/Tensordot:output:0#attention_score/ExpandDims:output:0*
T0*+
_output_shapes
:?????????2
attention_score/MatMul}
attention_score/ShapeShapeattention_score/MatMul:output:0*
T0*
_output_shapes
:2
attention_score/Shape?
attention_score/SqueezeSqueezeattention_score/MatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims

?????????2
attention_score/Squeeze?
attention_weight/SoftmaxSoftmax attention_score/Squeeze:output:0*
T0*'
_output_shapes
:?????????2
attention_weight/Softmax?
context_vector/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
context_vector/ExpandDims/dim?
context_vector/ExpandDims
ExpandDims"attention_weight/Softmax:softmax:0&context_vector/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
context_vector/ExpandDims?
context_vector/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
context_vector/transpose/perm?
context_vector/transpose	Transposelstm/transpose_1:y:0&context_vector/transpose/perm:output:0*
T0*+
_output_shapes
:????????? 2
context_vector/transpose?
context_vector/MatMulBatchMatMulV2context_vector/transpose:y:0"context_vector/ExpandDims:output:0*
T0*+
_output_shapes
:????????? 2
context_vector/MatMulz
context_vector/ShapeShapecontext_vector/MatMul:output:0*
T0*
_output_shapes
:2
context_vector/Shape?
context_vector/SqueezeSqueezecontext_vector/MatMul:output:0*
T0*'
_output_shapes
:????????? *
squeeze_dims

?????????2
context_vector/Squeeze~
attention_output/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
attention_output/concat/axis?
attention_output/concatConcatV2context_vector/Squeeze:output:0(last_hidden_state/strided_slice:output:0%attention_output/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
attention_output/concat?
&attention_vector/MatMul/ReadVariableOpReadVariableOp/attention_vector_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02(
&attention_vector/MatMul/ReadVariableOp?
attention_vector/MatMulMatMul attention_output/concat:output:0.attention_vector/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
attention_vector/MatMul?
attention_vector/TanhTanh!attention_vector/MatMul:product:0*
T0*(
_output_shapes
:??????????2
attention_vector/Tanhs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMulattention_vector/Tanh:y:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mulw
dropout/dropout/ShapeShapeattention_vector/Tanh:y:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed?2.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_2/Relu{
IdentityIdentitydense_2/Relu:activations:0^lstm/while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????:::::::::::::::2

lstm/while
lstm/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_18795

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed?2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_17463
model_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmodel_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_174222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????:::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_namemodel_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_16762

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_19656

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?1
?
while_body_19226
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*lstm_cell_matmul_readvariableop_resource_00
,lstm_cell_matmul_1_readvariableop_resource_0/
+lstm_cell_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/MatMul/ReadVariableOpReadVariableOp*lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identity_4k

Identity_5Identitylstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"X
)lstm_cell_biasadd_readvariableop_resource+lstm_cell_biasadd_readvariableop_resource_0"Z
*lstm_cell_matmul_1_readvariableop_resource,lstm_cell_matmul_1_readvariableop_resource_0"V
(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
g
K__inference_attention_weight_layer_call_and_return_conditional_losses_16680

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_15617

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
v
0__inference_attention_vector_layer_call_fn_19618

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*(
_output_shapes
:??????????*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_attention_vector_layer_call_and_return_conditional_losses_167332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????@:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: 
?
|
'__inference_dense_2_layer_call_fn_19685

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_168132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?D
?
?__inference_lstm_layer_call_and_return_conditional_losses_16224

inputs
lstm_cell_16142
lstm_cell_16144
lstm_cell_16146
identity??!lstm_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_16142lstm_cell_16144lstm_cell_16146*
Tin

2*
Tout
2*M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_157292#
!lstm_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_16142lstm_cell_16144lstm_cell_16146*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_16155*
condR
while_cond_16154*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0"^lstm_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????@:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_17422

inputs
model_17379
model_17381
model_17383
model_17385
model_17387
model_17389
model_17391
model_17393
model_17395
model_17397
model_17399
model_17401
model_17403
model_17405
model_17407
dense_17410
dense_17412
dense_1_17416
dense_1_17418
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?model/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_17379model_17381model_17383model_17385model_17387model_17389model_17391model_17393model_17395model_17397model_17399model_17401model_17403model_17405model_17407*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*1
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_170152
model/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0dense_17410dense_17412*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_171642
dense/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_171972
dropout_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_17416dense_1_17418*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_172212!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^model/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????:::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
?
?
while_cond_19225
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_19225___redundant_placeholder0-
)while_cond_19225___redundant_placeholder1-
)while_cond_19225___redundant_placeholder2-
)while_cond_19225___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?1
?
while_body_16473
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*lstm_cell_matmul_readvariableop_resource_00
,lstm_cell_matmul_1_readvariableop_resource_0/
+lstm_cell_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/MatMul/ReadVariableOpReadVariableOp*lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identity_4k

Identity_5Identitylstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"X
)lstm_cell_biasadd_readvariableop_resource+lstm_cell_biasadd_readvariableop_resource_0"Z
*lstm_cell_matmul_1_readvariableop_resource,lstm_cell_matmul_1_readvariableop_resource_0"V
(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_15729

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:????????? 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:????????? 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:????????? 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:????????? 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????@:????????? :????????? ::::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
N__inference_attention_score_vec_layer_call_and_return_conditional_losses_19513

inputs%
!tensordot_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
	Tensordotj
IdentityIdentityTensordot:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
while_body_16155
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
lstm_cell_16179_0
lstm_cell_16181_0
lstm_cell_16183_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
lstm_cell_16179
lstm_cell_16181
lstm_cell_16183??!lstm_cell/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3lstm_cell_16179_0lstm_cell_16181_0lstm_cell_16183_0*
Tin

2*
Tout
2*M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_157292#
!lstm_cell/StatefulPartitionedCall?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder*lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1p
IdentityIdentity	add_1:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1r

Identity_2Identityadd:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity*lstm_cell/StatefulPartitionedCall:output:1"^lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_4?

Identity_5Identity*lstm_cell/StatefulPartitionedCall:output:2"^lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"$
lstm_cell_16179lstm_cell_16179_0"$
lstm_cell_16181lstm_cell_16181_0"$
lstm_cell_16183lstm_cell_16183_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
|
'__inference_dense_1_layer_call_fn_19665

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_167862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_16757

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed?2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_16154
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_16154___redundant_placeholder0-
)while_cond_16154___redundant_placeholder1-
)while_cond_16154___redundant_placeholder2-
)while_cond_16154___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?A
?
@__inference_model_layer_call_and_return_conditional_losses_16931

inputs
conv1d_16885
conv1d_16887
conv1d_1_16890
conv1d_1_16892
conv1d_2_16895
conv1d_2_16897

lstm_16901

lstm_16903

lstm_16905
attention_score_vec_16909
attention_vector_16916
dense_1_16920
dense_1_16922
dense_2_16925
dense_2_16927
identity??+attention_score_vec/StatefulPartitionedCall?(attention_vector/StatefulPartitionedCall?conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?lstm/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_16885conv1d_16887*
Tin
2*
Tout
2*+
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_155402 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_16890conv1d_1_16892*
Tin
2*
Tout
2*+
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_155692"
 conv1d_1/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_16895conv1d_2_16897*
Tin
2*
Tout
2*+
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_155982"
 conv1d_2/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_156172
max_pooling1d/PartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0
lstm_16901
lstm_16903
lstm_16905*
Tin
2*
Tout
2*+
_output_shapes
:????????? *%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_164052
lstm/StatefulPartitionedCall?
!last_hidden_state/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_last_hidden_state_layer_call_and_return_conditional_losses_165962#
!last_hidden_state/PartitionedCall?
+attention_score_vec/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0attention_score_vec_16909*
Tin
2*
Tout
2*+
_output_shapes
:????????? *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_attention_score_vec_layer_call_and_return_conditional_losses_166442-
+attention_score_vec/StatefulPartitionedCall?
attention_score/PartitionedCallPartitionedCall4attention_score_vec/StatefulPartitionedCall:output:0*last_hidden_state/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_attention_score_layer_call_and_return_conditional_losses_166662!
attention_score/PartitionedCall?
 attention_weight/PartitionedCallPartitionedCall(attention_score/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_attention_weight_layer_call_and_return_conditional_losses_166802"
 attention_weight/PartitionedCall?
context_vector/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0)attention_weight/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_context_vector_layer_call_and_return_conditional_losses_167002 
context_vector/PartitionedCall?
 attention_output/PartitionedCallPartitionedCall'context_vector/PartitionedCall:output:0*last_hidden_state/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_attention_output_layer_call_and_return_conditional_losses_167162"
 attention_output/PartitionedCall?
(attention_vector/StatefulPartitionedCallStatefulPartitionedCall)attention_output/PartitionedCall:output:0attention_vector_16916*
Tin
2*
Tout
2*(
_output_shapes
:??????????*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_attention_vector_layer_call_and_return_conditional_losses_167332*
(attention_vector/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall1attention_vector/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_167572!
dropout/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_16920dense_1_16922*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_167862!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_16925dense_2_16927*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_168132!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^attention_score_vec/StatefulPartitionedCall)^attention_vector/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????:::::::::::::::2Z
+attention_score_vec/StatefulPartitionedCall+attention_score_vec/StatefulPartitionedCall2T
(attention_vector/StatefulPartitionedCall(attention_vector/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?O
?
__inference__traced_save_19920
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop/
+savev2_dense_1_1_kernel_read_readvariableop-
)savev2_dense_1_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop4
0savev2_lstm_lstm_cell_kernel_read_readvariableop>
:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop2
.savev2_lstm_lstm_cell_bias_read_readvariableop9
5savev2_attention_score_vec_kernel_read_readvariableop6
2savev2_attention_vector_kernel_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop6
2savev2_adam_dense_1_1_kernel_m_read_readvariableop4
0savev2_adam_dense_1_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop6
2savev2_adam_dense_1_1_kernel_v_read_readvariableop4
0savev2_adam_dense_1_1_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_9b4123b43e3b4ccb9b62d99388a0669a/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop+savev2_dense_1_1_kernel_read_readvariableop)savev2_dense_1_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop0savev2_lstm_lstm_cell_kernel_read_readvariableop:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop.savev2_lstm_lstm_cell_bias_read_readvariableop5savev2_attention_score_vec_kernel_read_readvariableop2savev2_attention_vector_kernel_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop2savev2_adam_dense_1_1_kernel_m_read_readvariableop0savev2_adam_dense_1_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop2savev2_adam_dense_1_1_kernel_v_read_readvariableop0savev2_adam_dense_1_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@ : : :: : : : : :@:@:@@:@:@@:@:	@?:	 ?:?:  :	@?:	?@:@:@@:@: : : : :@ : : ::@ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :(
$
"
_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:%!

_output_shapes
:	@?:%!

_output_shapes
:	 ?:!

_output_shapes	
:?:$ 

_output_shapes

:  :%!

_output_shapes
:	@?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: :  

_output_shapes
::$! 

_output_shapes

:@ : "

_output_shapes
: :$# 

_output_shapes

: : $

_output_shapes
::%

_output_shapes
: 
?
?
while_cond_19050
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_19050___redundant_placeholder0-
)while_cond_19050___redundant_placeholder1-
)while_cond_19050___redundant_placeholder2-
)while_cond_19050___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19718

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:????????? 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:????????? 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:????????? 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:????????? 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????@:????????? :????????? ::::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_15569

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingso
PadPadinputsPad/paddings:output:0*
T0*4
_output_shapes"
 :??????????????????@2
Padp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????@2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????@:::\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
h
L__inference_last_hidden_state_layer_call_and_return_conditional_losses_19528

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
$__inference_lstm_layer_call_fn_19147
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*4
_output_shapes"
 :?????????????????? *%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_160922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????@:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_18800

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_16786

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?V
?
?__inference_lstm_layer_call_and_return_conditional_losses_16558

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1?
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_16473*
condR
while_cond_16472*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::2
whilewhile:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
%__inference_model_layer_call_fn_17048
conv1d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*1
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_170152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_lstm_layer_call_fn_19486

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*+
_output_shapes
:????????? *%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_165582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?V
?
?__inference_lstm_layer_call_and_return_conditional_losses_16405

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1?
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_16320*
condR
while_cond_16319*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::2
whilewhile:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
M
1__inference_last_hidden_state_layer_call_fn_19541

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_last_hidden_state_layer_call_and_return_conditional_losses_165962
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
}
(__inference_conv1d_1_layer_call_fn_15579

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_155692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?2
?
model_lstm_while_body_17919!
model_lstm_while_loop_counter'
#model_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3 
model_lstm_strided_slice_1_0\
Xtensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensor_0.
*lstm_cell_matmul_readvariableop_resource_00
,lstm_cell_matmul_1_readvariableop_resource_0/
+lstm_cell_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
model_lstm_strided_slice_1Z
Vtensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensor,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemXtensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/MatMul/ReadVariableOpReadVariableOp*lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yi
add_1AddV2model_lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityj

Identity_1Identity#model_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identity_4k

Identity_5Identitylstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"X
)lstm_cell_biasadd_readvariableop_resource+lstm_cell_biasadd_readvariableop_resource_0"Z
*lstm_cell_matmul_1_readvariableop_resource,lstm_cell_matmul_1_readvariableop_resource_0"V
(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_readvariableop_resource_0":
model_lstm_strided_slice_1model_lstm_strided_slice_1_0"?
Vtensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensorXtensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
h
L__inference_last_hidden_state_layer_call_and_return_conditional_losses_16596

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
y
3__inference_attention_score_vec_layer_call_fn_19520

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:????????? *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_attention_score_vec_layer_call_and_return_conditional_losses_166442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: 
?V
?
?__inference_lstm_layer_call_and_return_conditional_losses_19464

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1?
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_19379*
condR
while_cond_19378*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::2
whilewhile:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
K__inference_attention_vector_layer_call_and_return_conditional_losses_19611

inputs"
matmul_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMulY
TanhTanhMatMul:product:0*
T0*(
_output_shapes
:??????????2
Tanh]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????@::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_17333

inputs
model_17290
model_17292
model_17294
model_17296
model_17298
model_17300
model_17302
model_17304
model_17306
model_17308
model_17310
model_17312
model_17314
model_17316
model_17318
dense_17321
dense_17323
dense_1_17327
dense_1_17329
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?model/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_17290model_17292model_17294model_17296model_17298model_17300model_17302model_17304model_17306model_17308model_17310model_17312model_17314model_17316model_17318*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*1
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_169312
model/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0dense_17321dense_17323*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_171642
dense/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_171922#
!dropout_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_17327dense_1_17329*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_172212!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^model/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????:::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
?
?
lstm_while_cond_18546
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_strided_slice_12
.lstm_while_cond_18546___redundant_placeholder02
.lstm_while_cond_18546___redundant_placeholder12
.lstm_while_cond_18546___redundant_placeholder22
.lstm_while_cond_18546___redundant_placeholder3
identity
]
LessLessplaceholderless_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
I
-__inference_max_pooling1d_layer_call_fn_15623

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_156172
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
lstm_while_cond_18279
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_strided_slice_12
.lstm_while_cond_18279___redundant_placeholder02
.lstm_while_cond_18279___redundant_placeholder12
.lstm_while_cond_18279___redundant_placeholder22
.lstm_while_cond_18279___redundant_placeholder3
identity
]
LessLessplaceholderless_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
z
%__inference_dense_layer_call_fn_18783

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_171642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
!__inference__traced_restore_20040
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias'
#assignvariableop_2_dense_1_1_kernel%
!assignvariableop_3_dense_1_1_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate$
 assignvariableop_9_conv1d_kernel#
assignvariableop_10_conv1d_bias'
#assignvariableop_11_conv1d_1_kernel%
!assignvariableop_12_conv1d_1_bias'
#assignvariableop_13_conv1d_2_kernel%
!assignvariableop_14_conv1d_2_bias-
)assignvariableop_15_lstm_lstm_cell_kernel7
3assignvariableop_16_lstm_lstm_cell_recurrent_kernel+
'assignvariableop_17_lstm_lstm_cell_bias2
.assignvariableop_18_attention_score_vec_kernel/
+assignvariableop_19_attention_vector_kernel&
"assignvariableop_20_dense_1_kernel$
 assignvariableop_21_dense_1_bias&
"assignvariableop_22_dense_2_kernel$
 assignvariableop_23_dense_2_bias
assignvariableop_24_total
assignvariableop_25_count
assignvariableop_26_total_1
assignvariableop_27_count_1+
'assignvariableop_28_adam_dense_kernel_m)
%assignvariableop_29_adam_dense_bias_m/
+assignvariableop_30_adam_dense_1_1_kernel_m-
)assignvariableop_31_adam_dense_1_1_bias_m+
'assignvariableop_32_adam_dense_kernel_v)
%assignvariableop_33_adam_dense_bias_v/
+assignvariableop_34_adam_dense_1_1_kernel_v-
)assignvariableop_35_adam_dense_1_1_bias_v
identity_37??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_1_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_1_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv1d_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_conv1d_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv1d_1_kernelIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv1d_1_biasIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv1d_2_kernelIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_conv1d_2_biasIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_lstm_lstm_cell_kernelIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp3assignvariableop_16_lstm_lstm_cell_recurrent_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp'assignvariableop_17_lstm_lstm_cell_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp.assignvariableop_18_attention_score_vec_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_attention_vector_kernelIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_1_kernelIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_1_biasIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_2_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_2_biasIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_totalIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_countIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_kernel_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp%assignvariableop_29_adam_dense_bias_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp+assignvariableop_30_adam_dense_1_1_kernel_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_1_1_bias_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_kernel_vIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp%assignvariableop_33_adam_dense_bias_vIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_adam_dense_1_1_kernel_vIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_1_1_bias_vIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_36?
Identity_37IdentityIdentity_36:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_37"#
identity_37Identity_37:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: 
?
?
while_body_16023
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
lstm_cell_16047_0
lstm_cell_16049_0
lstm_cell_16051_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
lstm_cell_16047
lstm_cell_16049
lstm_cell_16051??!lstm_cell/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3lstm_cell_16047_0lstm_cell_16049_0lstm_cell_16051_0*
Tin

2*
Tout
2*M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_156962#
!lstm_cell/StatefulPartitionedCall?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder*lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1p
IdentityIdentity	add_1:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1r

Identity_2Identityadd:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity*lstm_cell/StatefulPartitionedCall:output:1"^lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_4?

Identity_5Identity*lstm_cell/StatefulPartitionedCall:output:2"^lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"$
lstm_cell_16047lstm_cell_16047_0"$
lstm_cell_16049lstm_cell_16049_0"$
lstm_cell_16051lstm_cell_16051_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
$__inference_lstm_layer_call_fn_19158
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*4
_output_shapes"
 :?????????????????? *%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_162242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????@:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?1
?
while_body_18898
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*lstm_cell_matmul_readvariableop_resource_00
,lstm_cell_matmul_1_readvariableop_resource_0/
+lstm_cell_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/MatMul/ReadVariableOpReadVariableOp*lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identity_4k

Identity_5Identitylstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"X
)lstm_cell_biasadd_readvariableop_resource+lstm_cell_biasadd_readvariableop_resource_0"Z
*lstm_cell_matmul_1_readvariableop_resource,lstm_cell_matmul_1_readvariableop_resource_0"V
(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
@__inference_dense_layer_call_and_return_conditional_losses_17164

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?A
?
@__inference_model_layer_call_and_return_conditional_losses_16830
conv1d_input
conv1d_16237
conv1d_16239
conv1d_1_16242
conv1d_1_16244
conv1d_2_16247
conv1d_2_16249

lstm_16581

lstm_16583

lstm_16585
attention_score_vec_16653
attention_vector_16742
dense_1_16797
dense_1_16799
dense_2_16824
dense_2_16826
identity??+attention_score_vec/StatefulPartitionedCall?(attention_vector/StatefulPartitionedCall?conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?lstm/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallconv1d_inputconv1d_16237conv1d_16239*
Tin
2*
Tout
2*+
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_155402 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_16242conv1d_1_16244*
Tin
2*
Tout
2*+
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_155692"
 conv1d_1/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_16247conv1d_2_16249*
Tin
2*
Tout
2*+
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_155982"
 conv1d_2/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_156172
max_pooling1d/PartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0
lstm_16581
lstm_16583
lstm_16585*
Tin
2*
Tout
2*+
_output_shapes
:????????? *%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_164052
lstm/StatefulPartitionedCall?
!last_hidden_state/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_last_hidden_state_layer_call_and_return_conditional_losses_165962#
!last_hidden_state/PartitionedCall?
+attention_score_vec/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0attention_score_vec_16653*
Tin
2*
Tout
2*+
_output_shapes
:????????? *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_attention_score_vec_layer_call_and_return_conditional_losses_166442-
+attention_score_vec/StatefulPartitionedCall?
attention_score/PartitionedCallPartitionedCall4attention_score_vec/StatefulPartitionedCall:output:0*last_hidden_state/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_attention_score_layer_call_and_return_conditional_losses_166662!
attention_score/PartitionedCall?
 attention_weight/PartitionedCallPartitionedCall(attention_score/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_attention_weight_layer_call_and_return_conditional_losses_166802"
 attention_weight/PartitionedCall?
context_vector/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0)attention_weight/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_context_vector_layer_call_and_return_conditional_losses_167002 
context_vector/PartitionedCall?
 attention_output/PartitionedCallPartitionedCall'context_vector/PartitionedCall:output:0*last_hidden_state/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_attention_output_layer_call_and_return_conditional_losses_167162"
 attention_output/PartitionedCall?
(attention_vector/StatefulPartitionedCallStatefulPartitionedCall)attention_output/PartitionedCall:output:0attention_vector_16742*
Tin
2*
Tout
2*(
_output_shapes
:??????????*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_attention_vector_layer_call_and_return_conditional_losses_167332*
(attention_vector/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall1attention_vector/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_167572!
dropout/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_16797dense_1_16799*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_167862!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_16824dense_2_16826*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_168132!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^attention_score_vec/StatefulPartitionedCall)^attention_vector/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????:::::::::::::::2Z
+attention_score_vec/StatefulPartitionedCall+attention_score_vec/StatefulPartitionedCall2T
(attention_vector/StatefulPartitionedCall(attention_vector/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
g
K__inference_attention_weight_layer_call_and_return_conditional_losses_19567

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?D
?
?__inference_lstm_layer_call_and_return_conditional_losses_16092

inputs
lstm_cell_16010
lstm_cell_16012
lstm_cell_16014
identity??!lstm_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_16010lstm_cell_16012lstm_cell_16014*
Tin

2*
Tout
2*M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_156962#
!lstm_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_16010lstm_cell_16012lstm_cell_16014*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_16023*
condR
while_cond_16022*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0"^lstm_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????@:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

u
I__inference_context_vector_layer_call_and_return_conditional_losses_19584
inputs_0
inputs_1
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputs_1ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*+
_output_shapes
:????????? 2
	transpose{
MatMulBatchMatMulV2transpose:y:0ExpandDims:output:0*
T0*+
_output_shapes
:????????? 2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shape?
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:????????? *
squeeze_dims

?????????2	
Squeezed
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:????????? :?????????:U Q
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_17192

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed?2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

s
I__inference_context_vector_layer_call_and_return_conditional_losses_16700

inputs
inputs_1
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputs_1ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:????????? 2
	transpose{
MatMulBatchMatMulV2transpose:y:0ExpandDims:output:0*
T0*+
_output_shapes
:????????? 2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shape?
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:????????? *
squeeze_dims

?????????2	
Squeezed
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:????????? :?????????:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_19630

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed?2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?2
?
model_lstm_while_body_17630!
model_lstm_while_loop_counter'
#model_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3 
model_lstm_strided_slice_1_0\
Xtensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensor_0.
*lstm_cell_matmul_readvariableop_resource_00
,lstm_cell_matmul_1_readvariableop_resource_0/
+lstm_cell_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
model_lstm_strided_slice_1Z
Vtensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensor,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemXtensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/MatMul/ReadVariableOpReadVariableOp*lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yi
add_1AddV2model_lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityj

Identity_1Identity#model_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identity_4k

Identity_5Identitylstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"X
)lstm_cell_biasadd_readvariableop_resource+lstm_cell_biasadd_readvariableop_resource_0"Z
*lstm_cell_matmul_1_readvariableop_resource,lstm_cell_matmul_1_readvariableop_resource_0"V
(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_readvariableop_resource_0":
model_lstm_strided_slice_1model_lstm_strided_slice_1_0"?
Vtensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensorXtensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
%__inference_model_layer_call_fn_18728

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*1
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_169312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?1
?
while_body_19379
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*lstm_cell_matmul_readvariableop_resource_00
,lstm_cell_matmul_1_readvariableop_resource_0/
+lstm_cell_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/MatMul/ReadVariableOpReadVariableOp*lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identity_4k

Identity_5Identitylstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"X
)lstm_cell_biasadd_readvariableop_resource+lstm_cell_biasadd_readvariableop_resource_0"Z
*lstm_cell_matmul_1_readvariableop_resource,lstm_cell_matmul_1_readvariableop_resource_0"V
(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_19635

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_16319
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_16319___redundant_placeholder0-
)while_cond_16319___redundant_placeholder1-
)while_cond_16319___redundant_placeholder2-
)while_cond_16319___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
h
L__inference_last_hidden_state_layer_call_and_return_conditional_losses_19536

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_17238
model_input
model_17122
model_17124
model_17126
model_17128
model_17130
model_17132
model_17134
model_17136
model_17138
model_17140
model_17142
model_17144
model_17146
model_17148
model_17150
dense_17175
dense_17177
dense_1_17232
dense_1_17234
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?model/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallmodel_inputmodel_17122model_17124model_17126model_17128model_17130model_17132model_17134model_17136model_17138model_17140model_17142model_17144model_17146model_17148model_17150*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*1
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_169312
model/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0dense_17175dense_17177*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_171642
dense/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_171922#
!dropout_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_17232dense_1_17234*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_172212!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^model/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????:::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_namemodel_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
?
L
0__inference_attention_weight_layer_call_fn_19572

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_attention_weight_layer_call_and_return_conditional_losses_166802
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?W
?
?__inference_lstm_layer_call_and_return_conditional_losses_19136
inputs_0,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1?
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_19051*
condR
while_cond_19050*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????@:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
v
J__inference_attention_score_layer_call_and_return_conditional_losses_19556
inputs_0
inputs_1
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputs_1ExpandDims/dim:output:0*
T0*+
_output_shapes
:????????? 2

ExpandDimsv
MatMulBatchMatMulV2inputs_0ExpandDims:output:0*
T0*+
_output_shapes
:?????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shape?
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims

?????????2	
Squeezed
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:????????? :????????? :U Q
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?
E
)__inference_dropout_1_layer_call_fn_18810

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_171972
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?V
?
?__inference_lstm_layer_call_and_return_conditional_losses_19311

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1?
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_19226*
condR
while_cond_19225*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::2
whilewhile:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
N__inference_attention_score_vec_layer_call_and_return_conditional_losses_16644

inputs%
!tensordot_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
	Tensordotj
IdentityIdentityTensordot:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: 
?2
?
lstm_while_body_18547
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_strided_slice_1_0V
Rtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0.
*lstm_cell_matmul_readvariableop_resource_00
,lstm_cell_matmul_1_readvariableop_resource_0/
+lstm_cell_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_strided_slice_1T
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/MatMul/ReadVariableOpReadVariableOp*lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yc
add_1AddV2lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityd

Identity_1Identitylstm_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identity_4k

Identity_5Identitylstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"X
)lstm_cell_biasadd_readvariableop_resource+lstm_cell_biasadd_readvariableop_resource_0"Z
*lstm_cell_matmul_1_readvariableop_resource,lstm_cell_matmul_1_readvariableop_resource_0"V
(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_readvariableop_resource_0".
lstm_strided_slice_1lstm_strided_slice_1_0"?
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
Z
.__inference_context_vector_layer_call_fn_19590
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_context_vector_layer_call_and_return_conditional_losses_167002
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:????????? :?????????:U Q
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
C__inference_conv1d_2_layer_call_and_return_conditional_losses_15598

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingso
PadPadinputsPad/paddings:output:0*
T0*4
_output_shapes"
 :??????????????????@2
Padp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????@2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????@:::\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
M
1__inference_last_hidden_state_layer_call_fn_19546

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_last_hidden_state_layer_call_and_return_conditional_losses_166042
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
h
L__inference_last_hidden_state_layer_call_and_return_conditional_losses_16604

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_16964
conv1d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*1
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_169312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
)__inference_lstm_cell_layer_call_fn_19785

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_157292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????@:????????? :????????? :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
%__inference_model_layer_call_fn_18763

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*1
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_170152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_16472
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_16472___redundant_placeholder0-
)while_cond_16472___redundant_placeholder1-
)while_cond_16472___redundant_placeholder2-
)while_cond_16472___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
w
K__inference_attention_output_layer_call_and_return_conditional_losses_19597
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :????????? :Q M
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?
?
$__inference_lstm_layer_call_fn_19475

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*+
_output_shapes
:????????? *%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_164052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?@
?
@__inference_model_layer_call_and_return_conditional_losses_16879
conv1d_input
conv1d_16833
conv1d_16835
conv1d_1_16838
conv1d_1_16840
conv1d_2_16843
conv1d_2_16845

lstm_16849

lstm_16851

lstm_16853
attention_score_vec_16857
attention_vector_16864
dense_1_16868
dense_1_16870
dense_2_16873
dense_2_16875
identity??+attention_score_vec/StatefulPartitionedCall?(attention_vector/StatefulPartitionedCall?conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?lstm/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallconv1d_inputconv1d_16833conv1d_16835*
Tin
2*
Tout
2*+
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_155402 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_16838conv1d_1_16840*
Tin
2*
Tout
2*+
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_155692"
 conv1d_1/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_16843conv1d_2_16845*
Tin
2*
Tout
2*+
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_155982"
 conv1d_2/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_156172
max_pooling1d/PartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0
lstm_16849
lstm_16851
lstm_16853*
Tin
2*
Tout
2*+
_output_shapes
:????????? *%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_165582
lstm/StatefulPartitionedCall?
!last_hidden_state/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_last_hidden_state_layer_call_and_return_conditional_losses_166042#
!last_hidden_state/PartitionedCall?
+attention_score_vec/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0attention_score_vec_16857*
Tin
2*
Tout
2*+
_output_shapes
:????????? *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_attention_score_vec_layer_call_and_return_conditional_losses_166442-
+attention_score_vec/StatefulPartitionedCall?
attention_score/PartitionedCallPartitionedCall4attention_score_vec/StatefulPartitionedCall:output:0*last_hidden_state/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_attention_score_layer_call_and_return_conditional_losses_166662!
attention_score/PartitionedCall?
 attention_weight/PartitionedCallPartitionedCall(attention_score/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_attention_weight_layer_call_and_return_conditional_losses_166802"
 attention_weight/PartitionedCall?
context_vector/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0)attention_weight/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_context_vector_layer_call_and_return_conditional_losses_167002 
context_vector/PartitionedCall?
 attention_output/PartitionedCallPartitionedCall'context_vector/PartitionedCall:output:0*last_hidden_state/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_attention_output_layer_call_and_return_conditional_losses_167162"
 attention_output/PartitionedCall?
(attention_vector/StatefulPartitionedCallStatefulPartitionedCall)attention_output/PartitionedCall:output:0attention_vector_16864*
Tin
2*
Tout
2*(
_output_shapes
:??????????*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_attention_vector_layer_call_and_return_conditional_losses_167332*
(attention_vector/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall1attention_vector/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_167622
dropout/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_16868dense_1_16870*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_167862!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_16873dense_2_16875*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_168132!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^attention_score_vec/StatefulPartitionedCall)^attention_vector/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????:::::::::::::::2Z
+attention_score_vec/StatefulPartitionedCall+attention_score_vec/StatefulPartitionedCall2T
(attention_vector/StatefulPartitionedCall(attention_vector/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_15696

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:????????? 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:????????? 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:????????? 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:????????? 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????@:????????? :????????? ::::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?4
?
(sequential_1_model_lstm_while_body_15360.
*sequential_1_model_lstm_while_loop_counter4
0sequential_1_model_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3-
)sequential_1_model_lstm_strided_slice_1_0i
etensorarrayv2read_tensorlistgetitem_sequential_1_model_lstm_tensorarrayunstack_tensorlistfromtensor_0.
*lstm_cell_matmul_readvariableop_resource_00
,lstm_cell_matmul_1_readvariableop_resource_0/
+lstm_cell_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5+
'sequential_1_model_lstm_strided_slice_1g
ctensorarrayv2read_tensorlistgetitem_sequential_1_model_lstm_tensorarrayunstack_tensorlistfromtensor,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemetensorarrayv2read_tensorlistgetitem_sequential_1_model_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/MatMul/ReadVariableOpReadVariableOp*lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yv
add_1AddV2*sequential_1_model_lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityw

Identity_1Identity0sequential_1_model_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identity_4k

Identity_5Identitylstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"X
)lstm_cell_biasadd_readvariableop_resource+lstm_cell_biasadd_readvariableop_resource_0"Z
*lstm_cell_matmul_1_readvariableop_resource,lstm_cell_matmul_1_readvariableop_resource_0"V
(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_readvariableop_resource_0"T
'sequential_1_model_lstm_strided_slice_1)sequential_1_model_lstm_strided_slice_1_0"?
ctensorarrayv2read_tensorlistgetitem_sequential_1_model_lstm_tensorarrayunstack_tensorlistfromtensoretensorarrayv2read_tensorlistgetitem_sequential_1_model_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?W
?
?__inference_lstm_layer_call_and_return_conditional_losses_18983
inputs_0,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1?
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_18898*
condR
while_cond_18897*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????@:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
t
J__inference_attention_score_layer_call_and_return_conditional_losses_16666

inputs
inputs_1
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputs_1ExpandDims/dim:output:0*
T0*+
_output_shapes
:????????? 2

ExpandDimst
MatMulBatchMatMulV2inputsExpandDims:output:0*
T0*+
_output_shapes
:?????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shape?
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims

?????????2	
Squeezed
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:????????? :????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?2
?
lstm_while_body_18280
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_strided_slice_1_0V
Rtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0.
*lstm_cell_matmul_readvariableop_resource_00
,lstm_cell_matmul_1_readvariableop_resource_0/
+lstm_cell_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_strided_slice_1T
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/MatMul/ReadVariableOpReadVariableOp*lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yc
add_1AddV2lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityd

Identity_1Identitylstm_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identity_4k

Identity_5Identitylstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"X
)lstm_cell_biasadd_readvariableop_resource+lstm_cell_biasadd_readvariableop_resource_0"Z
*lstm_cell_matmul_1_readvariableop_resource,lstm_cell_matmul_1_readvariableop_resource_0"V
(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_readvariableop_resource_0".
lstm_strided_slice_1lstm_strided_slice_1_0"?
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
,__inference_sequential_1_layer_call_fn_18166

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_174222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????:::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
?
?
,__inference_sequential_1_layer_call_fn_18123

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_173332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????:::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
?
?
@__inference_dense_layer_call_and_return_conditional_losses_18774

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
)__inference_lstm_cell_layer_call_fn_19768

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_156962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????@:????????? :????????? :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
[
/__inference_attention_score_layer_call_fn_19562
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_attention_score_layer_call_and_return_conditional_losses_166662
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:????????? :????????? :U Q
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?
?
model_lstm_while_cond_17918!
model_lstm_while_loop_counter'
#model_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3#
less_model_lstm_strided_slice_18
4model_lstm_while_cond_17918___redundant_placeholder08
4model_lstm_while_cond_17918___redundant_placeholder18
4model_lstm_while_cond_17918___redundant_placeholder28
4model_lstm_while_cond_17918___redundant_placeholder3
identity
c
LessLessplaceholderless_model_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?1
?
while_body_19051
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*lstm_cell_matmul_readvariableop_resource_00
,lstm_cell_matmul_1_readvariableop_resource_0/
+lstm_cell_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/MatMul/ReadVariableOpReadVariableOp*lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identity_4k

Identity_5Identitylstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"X
)lstm_cell_biasadd_readvariableop_resource+lstm_cell_biasadd_readvariableop_resource_0"Z
*lstm_cell_matmul_1_readvariableop_resource,lstm_cell_matmul_1_readvariableop_resource_0"V
(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
B__inference_dense_2_layer_call_and_return_conditional_losses_16813

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_19378
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_19378___redundant_placeholder0-
)while_cond_19378___redundant_placeholder1-
)while_cond_19378___redundant_placeholder2-
)while_cond_19378___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?	
?
(sequential_1_model_lstm_while_cond_15359.
*sequential_1_model_lstm_while_loop_counter4
0sequential_1_model_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_30
,less_sequential_1_model_lstm_strided_slice_1E
Asequential_1_model_lstm_while_cond_15359___redundant_placeholder0E
Asequential_1_model_lstm_while_cond_15359___redundant_placeholder1E
Asequential_1_model_lstm_while_cond_15359___redundant_placeholder2E
Asequential_1_model_lstm_while_cond_15359___redundant_placeholder3
identity
p
LessLessplaceholder,less_sequential_1_model_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
b
)__inference_dropout_1_layer_call_fn_18805

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_171922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
u
K__inference_attention_output_layer_call_and_return_conditional_losses_16716

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
}
(__inference_conv1d_2_layer_call_fn_15608

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_155982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_dense_2_layer_call_and_return_conditional_losses_19676

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
#__inference_signature_wrapper_17516
model_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmodel_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__wrapped_model_155212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????:::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_namemodel_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
?
?
while_cond_18897
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_18897___redundant_placeholder0-
)while_cond_18897___redundant_placeholder1-
)while_cond_18897___redundant_placeholder2-
)while_cond_18897___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
??
?
@__inference_model_layer_call_and_return_conditional_losses_18693

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource1
-lstm_lstm_cell_matmul_readvariableop_resource3
/lstm_lstm_cell_matmul_1_readvariableop_resource2
.lstm_lstm_cell_biasadd_readvariableop_resource9
5attention_score_vec_tensordot_readvariableop_resource3
/attention_vector_matmul_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??
lstm/while?
conv1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d/Pad/paddings{

conv1d/PadPadinputsconv1d/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2

conv1d/Pad~
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimsconv1d/Pad:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d/Relu?
conv1d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_1/Pad/paddings?
conv1d_1/PadPadconv1d/Relu:activations:0conv1d_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_1/Pad?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d_1/Pad:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_1/Relu?
conv1d_2/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_2/Pad/paddings?
conv1d_2/PadPadconv1d_1/Relu:activations:0conv1d_2/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_2/Pad?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimsconv1d_2/Pad:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_2/Relu~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dim?
max_pooling1d/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0%max_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
max_pooling1d/ExpandDims?
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPool?
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
max_pooling1d/Squeezef

lstm/ShapeShapemax_pooling1d/Squeeze:output:0*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack?
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1?
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2?
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros/mul/y?
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros/packed/1?
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const?

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros_1/mul/y?
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/zeros_1/Less/y?
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm/zeros_1/packed/1?
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const?
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm?
lstm/transpose	Transposemax_pooling1d/Squeeze:output:0lstm/transpose/perm:output:0*
T0*+
_output_shapes
:?????????@2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1?
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack?
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1?
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2?
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1?
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm/TensorArrayV2/element_shape?
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2?
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape?
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor?
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack?
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1?
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2?
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
lstm/strided_slice_2?
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOp?
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_cell/MatMul?
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOp?
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_cell/MatMul_1?
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_cell/add?
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOp?
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_cell/BiasAddn
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell/Const?
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim?
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm/lstm_cell/split?
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/Sigmoid?
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/Sigmoid_1?
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/mul?
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/Relu?
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/mul_1?
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/add_1?
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/Sigmoid_2?
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/Relu_1?
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm/lstm_cell/mul_2?
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2$
"lstm/TensorArrayV2_1/element_shape?
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time?
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counter?

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
lstm_while_body_18547*!
condR
lstm_while_cond_18546*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2

lstm/while?
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    27
5lstm/TensorArrayV2Stack/TensorListStack/element_shape?
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack?
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm/strided_slice_3/stack?
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1?
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2?
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
lstm/strided_slice_3?
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm?
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtime?
%last_hidden_state/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2'
%last_hidden_state/strided_slice/stack?
'last_hidden_state/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2)
'last_hidden_state/strided_slice/stack_1?
'last_hidden_state/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2)
'last_hidden_state/strided_slice/stack_2?
last_hidden_state/strided_sliceStridedSlicelstm/transpose_1:y:0.last_hidden_state/strided_slice/stack:output:00last_hidden_state/strided_slice/stack_1:output:00last_hidden_state/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *

begin_mask*
end_mask*
shrink_axis_mask2!
last_hidden_state/strided_slice?
,attention_score_vec/Tensordot/ReadVariableOpReadVariableOp5attention_score_vec_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02.
,attention_score_vec/Tensordot/ReadVariableOp?
"attention_score_vec/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"attention_score_vec/Tensordot/axes?
"attention_score_vec/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"attention_score_vec/Tensordot/free?
#attention_score_vec/Tensordot/ShapeShapelstm/transpose_1:y:0*
T0*
_output_shapes
:2%
#attention_score_vec/Tensordot/Shape?
+attention_score_vec/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+attention_score_vec/Tensordot/GatherV2/axis?
&attention_score_vec/Tensordot/GatherV2GatherV2,attention_score_vec/Tensordot/Shape:output:0+attention_score_vec/Tensordot/free:output:04attention_score_vec/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&attention_score_vec/Tensordot/GatherV2?
-attention_score_vec/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-attention_score_vec/Tensordot/GatherV2_1/axis?
(attention_score_vec/Tensordot/GatherV2_1GatherV2,attention_score_vec/Tensordot/Shape:output:0+attention_score_vec/Tensordot/axes:output:06attention_score_vec/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(attention_score_vec/Tensordot/GatherV2_1?
#attention_score_vec/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#attention_score_vec/Tensordot/Const?
"attention_score_vec/Tensordot/ProdProd/attention_score_vec/Tensordot/GatherV2:output:0,attention_score_vec/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"attention_score_vec/Tensordot/Prod?
%attention_score_vec/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%attention_score_vec/Tensordot/Const_1?
$attention_score_vec/Tensordot/Prod_1Prod1attention_score_vec/Tensordot/GatherV2_1:output:0.attention_score_vec/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$attention_score_vec/Tensordot/Prod_1?
)attention_score_vec/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)attention_score_vec/Tensordot/concat/axis?
$attention_score_vec/Tensordot/concatConcatV2+attention_score_vec/Tensordot/free:output:0+attention_score_vec/Tensordot/axes:output:02attention_score_vec/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$attention_score_vec/Tensordot/concat?
#attention_score_vec/Tensordot/stackPack+attention_score_vec/Tensordot/Prod:output:0-attention_score_vec/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#attention_score_vec/Tensordot/stack?
'attention_score_vec/Tensordot/transpose	Transposelstm/transpose_1:y:0-attention_score_vec/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2)
'attention_score_vec/Tensordot/transpose?
%attention_score_vec/Tensordot/ReshapeReshape+attention_score_vec/Tensordot/transpose:y:0,attention_score_vec/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%attention_score_vec/Tensordot/Reshape?
$attention_score_vec/Tensordot/MatMulMatMul.attention_score_vec/Tensordot/Reshape:output:04attention_score_vec/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2&
$attention_score_vec/Tensordot/MatMul?
%attention_score_vec/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2'
%attention_score_vec/Tensordot/Const_2?
+attention_score_vec/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+attention_score_vec/Tensordot/concat_1/axis?
&attention_score_vec/Tensordot/concat_1ConcatV2/attention_score_vec/Tensordot/GatherV2:output:0.attention_score_vec/Tensordot/Const_2:output:04attention_score_vec/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&attention_score_vec/Tensordot/concat_1?
attention_score_vec/TensordotReshape.attention_score_vec/Tensordot/MatMul:product:0/attention_score_vec/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
attention_score_vec/Tensordot?
attention_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
attention_score/ExpandDims/dim?
attention_score/ExpandDims
ExpandDims(last_hidden_state/strided_slice:output:0'attention_score/ExpandDims/dim:output:0*
T0*+
_output_shapes
:????????? 2
attention_score/ExpandDims?
attention_score/MatMulBatchMatMulV2&attention_score_vec/Tensordot:output:0#attention_score/ExpandDims:output:0*
T0*+
_output_shapes
:?????????2
attention_score/MatMul}
attention_score/ShapeShapeattention_score/MatMul:output:0*
T0*
_output_shapes
:2
attention_score/Shape?
attention_score/SqueezeSqueezeattention_score/MatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims

?????????2
attention_score/Squeeze?
attention_weight/SoftmaxSoftmax attention_score/Squeeze:output:0*
T0*'
_output_shapes
:?????????2
attention_weight/Softmax?
context_vector/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
context_vector/ExpandDims/dim?
context_vector/ExpandDims
ExpandDims"attention_weight/Softmax:softmax:0&context_vector/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
context_vector/ExpandDims?
context_vector/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
context_vector/transpose/perm?
context_vector/transpose	Transposelstm/transpose_1:y:0&context_vector/transpose/perm:output:0*
T0*+
_output_shapes
:????????? 2
context_vector/transpose?
context_vector/MatMulBatchMatMulV2context_vector/transpose:y:0"context_vector/ExpandDims:output:0*
T0*+
_output_shapes
:????????? 2
context_vector/MatMulz
context_vector/ShapeShapecontext_vector/MatMul:output:0*
T0*
_output_shapes
:2
context_vector/Shape?
context_vector/SqueezeSqueezecontext_vector/MatMul:output:0*
T0*'
_output_shapes
:????????? *
squeeze_dims

?????????2
context_vector/Squeeze~
attention_output/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
attention_output/concat/axis?
attention_output/concatConcatV2context_vector/Squeeze:output:0(last_hidden_state/strided_slice:output:0%attention_output/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
attention_output/concat?
&attention_vector/MatMul/ReadVariableOpReadVariableOp/attention_vector_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02(
&attention_vector/MatMul/ReadVariableOp?
attention_vector/MatMulMatMul attention_output/concat:output:0.attention_vector/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
attention_vector/MatMul?
attention_vector/TanhTanh!attention_vector/MatMul:product:0*
T0*(
_output_shapes
:??????????2
attention_vector/Tanh~
dropout/IdentityIdentityattention_vector/Tanh:y:0*
T0*(
_output_shapes
:??????????2
dropout/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_2/Relu{
IdentityIdentitydense_2/Relu:activations:0^lstm/while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????:::::::::::::::2

lstm/while
lstm/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_17197

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_17284
model_input
model_17241
model_17243
model_17245
model_17247
model_17249
model_17251
model_17253
model_17255
model_17257
model_17259
model_17261
model_17263
model_17265
model_17267
model_17269
dense_17272
dense_17274
dense_1_17278
dense_1_17280
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?model/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallmodel_inputmodel_17241model_17243model_17245model_17247model_17249model_17251model_17253model_17255model_17257model_17259model_17261model_17263model_17265model_17267model_17269*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*1
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_170152
model/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0dense_17272dense_17274*
Tin
2*
Tout
2*'
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_171642
dense/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_171972
dropout_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_17278dense_1_17280*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_172212!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^model/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????:::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_namemodel_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
??
?

 __inference__wrapped_model_15521
model_inputI
Esequential_1_model_conv1d_conv1d_expanddims_1_readvariableop_resource=
9sequential_1_model_conv1d_biasadd_readvariableop_resourceK
Gsequential_1_model_conv1d_1_conv1d_expanddims_1_readvariableop_resource?
;sequential_1_model_conv1d_1_biasadd_readvariableop_resourceK
Gsequential_1_model_conv1d_2_conv1d_expanddims_1_readvariableop_resource?
;sequential_1_model_conv1d_2_biasadd_readvariableop_resourceD
@sequential_1_model_lstm_lstm_cell_matmul_readvariableop_resourceF
Bsequential_1_model_lstm_lstm_cell_matmul_1_readvariableop_resourceE
Asequential_1_model_lstm_lstm_cell_biasadd_readvariableop_resourceL
Hsequential_1_model_attention_score_vec_tensordot_readvariableop_resourceF
Bsequential_1_model_attention_vector_matmul_readvariableop_resource=
9sequential_1_model_dense_1_matmul_readvariableop_resource>
:sequential_1_model_dense_1_biasadd_readvariableop_resource=
9sequential_1_model_dense_2_matmul_readvariableop_resource>
:sequential_1_model_dense_2_biasadd_readvariableop_resource5
1sequential_1_dense_matmul_readvariableop_resource6
2sequential_1_dense_biasadd_readvariableop_resource7
3sequential_1_dense_1_matmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resource
identity??sequential_1/model/lstm/while?
&sequential_1/model/conv1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2(
&sequential_1/model/conv1d/Pad/paddings?
sequential_1/model/conv1d/PadPadmodel_input/sequential_1/model/conv1d/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2
sequential_1/model/conv1d/Pad?
/sequential_1/model/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_1/model/conv1d/conv1d/ExpandDims/dim?
+sequential_1/model/conv1d/conv1d/ExpandDims
ExpandDims&sequential_1/model/conv1d/Pad:output:08sequential_1/model/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2-
+sequential_1/model/conv1d/conv1d/ExpandDims?
<sequential_1/model/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEsequential_1_model_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<sequential_1/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
1sequential_1/model/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_1/model/conv1d/conv1d/ExpandDims_1/dim?
-sequential_1/model/conv1d/conv1d/ExpandDims_1
ExpandDimsDsequential_1/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0:sequential_1/model/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-sequential_1/model/conv1d/conv1d/ExpandDims_1?
 sequential_1/model/conv1d/conv1dConv2D4sequential_1/model/conv1d/conv1d/ExpandDims:output:06sequential_1/model/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2"
 sequential_1/model/conv1d/conv1d?
(sequential_1/model/conv1d/conv1d/SqueezeSqueeze)sequential_1/model/conv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2*
(sequential_1/model/conv1d/conv1d/Squeeze?
0sequential_1/model/conv1d/BiasAdd/ReadVariableOpReadVariableOp9sequential_1_model_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0sequential_1/model/conv1d/BiasAdd/ReadVariableOp?
!sequential_1/model/conv1d/BiasAddBiasAdd1sequential_1/model/conv1d/conv1d/Squeeze:output:08sequential_1/model/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2#
!sequential_1/model/conv1d/BiasAdd?
sequential_1/model/conv1d/ReluRelu*sequential_1/model/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2 
sequential_1/model/conv1d/Relu?
(sequential_1/model/conv1d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2*
(sequential_1/model/conv1d_1/Pad/paddings?
sequential_1/model/conv1d_1/PadPad,sequential_1/model/conv1d/Relu:activations:01sequential_1/model/conv1d_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2!
sequential_1/model/conv1d_1/Pad?
1sequential_1/model/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_1/model/conv1d_1/conv1d/ExpandDims/dim?
-sequential_1/model/conv1d_1/conv1d/ExpandDims
ExpandDims(sequential_1/model/conv1d_1/Pad:output:0:sequential_1/model/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2/
-sequential_1/model/conv1d_1/conv1d/ExpandDims?
>sequential_1/model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpGsequential_1_model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02@
>sequential_1/model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
3sequential_1/model/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3sequential_1/model/conv1d_1/conv1d/ExpandDims_1/dim?
/sequential_1/model/conv1d_1/conv1d/ExpandDims_1
ExpandDimsFsequential_1/model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0<sequential_1/model/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@21
/sequential_1/model/conv1d_1/conv1d/ExpandDims_1?
"sequential_1/model/conv1d_1/conv1dConv2D6sequential_1/model/conv1d_1/conv1d/ExpandDims:output:08sequential_1/model/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2$
"sequential_1/model/conv1d_1/conv1d?
*sequential_1/model/conv1d_1/conv1d/SqueezeSqueeze+sequential_1/model/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2,
*sequential_1/model/conv1d_1/conv1d/Squeeze?
2sequential_1/model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp;sequential_1_model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2sequential_1/model/conv1d_1/BiasAdd/ReadVariableOp?
#sequential_1/model/conv1d_1/BiasAddBiasAdd3sequential_1/model/conv1d_1/conv1d/Squeeze:output:0:sequential_1/model/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2%
#sequential_1/model/conv1d_1/BiasAdd?
 sequential_1/model/conv1d_1/ReluRelu,sequential_1/model/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2"
 sequential_1/model/conv1d_1/Relu?
(sequential_1/model/conv1d_2/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2*
(sequential_1/model/conv1d_2/Pad/paddings?
sequential_1/model/conv1d_2/PadPad.sequential_1/model/conv1d_1/Relu:activations:01sequential_1/model/conv1d_2/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2!
sequential_1/model/conv1d_2/Pad?
1sequential_1/model/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_1/model/conv1d_2/conv1d/ExpandDims/dim?
-sequential_1/model/conv1d_2/conv1d/ExpandDims
ExpandDims(sequential_1/model/conv1d_2/Pad:output:0:sequential_1/model/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2/
-sequential_1/model/conv1d_2/conv1d/ExpandDims?
>sequential_1/model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpGsequential_1_model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02@
>sequential_1/model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
3sequential_1/model/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3sequential_1/model/conv1d_2/conv1d/ExpandDims_1/dim?
/sequential_1/model/conv1d_2/conv1d/ExpandDims_1
ExpandDimsFsequential_1/model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0<sequential_1/model/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@21
/sequential_1/model/conv1d_2/conv1d/ExpandDims_1?
"sequential_1/model/conv1d_2/conv1dConv2D6sequential_1/model/conv1d_2/conv1d/ExpandDims:output:08sequential_1/model/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2$
"sequential_1/model/conv1d_2/conv1d?
*sequential_1/model/conv1d_2/conv1d/SqueezeSqueeze+sequential_1/model/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2,
*sequential_1/model/conv1d_2/conv1d/Squeeze?
2sequential_1/model/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp;sequential_1_model_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2sequential_1/model/conv1d_2/BiasAdd/ReadVariableOp?
#sequential_1/model/conv1d_2/BiasAddBiasAdd3sequential_1/model/conv1d_2/conv1d/Squeeze:output:0:sequential_1/model/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2%
#sequential_1/model/conv1d_2/BiasAdd?
 sequential_1/model/conv1d_2/ReluRelu,sequential_1/model/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2"
 sequential_1/model/conv1d_2/Relu?
/sequential_1/model/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_1/model/max_pooling1d/ExpandDims/dim?
+sequential_1/model/max_pooling1d/ExpandDims
ExpandDims.sequential_1/model/conv1d_2/Relu:activations:08sequential_1/model/max_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2-
+sequential_1/model/max_pooling1d/ExpandDims?
(sequential_1/model/max_pooling1d/MaxPoolMaxPool4sequential_1/model/max_pooling1d/ExpandDims:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2*
(sequential_1/model/max_pooling1d/MaxPool?
(sequential_1/model/max_pooling1d/SqueezeSqueeze1sequential_1/model/max_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2*
(sequential_1/model/max_pooling1d/Squeeze?
sequential_1/model/lstm/ShapeShape1sequential_1/model/max_pooling1d/Squeeze:output:0*
T0*
_output_shapes
:2
sequential_1/model/lstm/Shape?
+sequential_1/model/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_1/model/lstm/strided_slice/stack?
-sequential_1/model/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_1/model/lstm/strided_slice/stack_1?
-sequential_1/model/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_1/model/lstm/strided_slice/stack_2?
%sequential_1/model/lstm/strided_sliceStridedSlice&sequential_1/model/lstm/Shape:output:04sequential_1/model/lstm/strided_slice/stack:output:06sequential_1/model/lstm/strided_slice/stack_1:output:06sequential_1/model/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_1/model/lstm/strided_slice?
#sequential_1/model/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential_1/model/lstm/zeros/mul/y?
!sequential_1/model/lstm/zeros/mulMul.sequential_1/model/lstm/strided_slice:output:0,sequential_1/model/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_1/model/lstm/zeros/mul?
$sequential_1/model/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_1/model/lstm/zeros/Less/y?
"sequential_1/model/lstm/zeros/LessLess%sequential_1/model/lstm/zeros/mul:z:0-sequential_1/model/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_1/model/lstm/zeros/Less?
&sequential_1/model/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_1/model/lstm/zeros/packed/1?
$sequential_1/model/lstm/zeros/packedPack.sequential_1/model/lstm/strided_slice:output:0/sequential_1/model/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/model/lstm/zeros/packed?
#sequential_1/model/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_1/model/lstm/zeros/Const?
sequential_1/model/lstm/zerosFill-sequential_1/model/lstm/zeros/packed:output:0,sequential_1/model/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
sequential_1/model/lstm/zeros?
%sequential_1/model/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/model/lstm/zeros_1/mul/y?
#sequential_1/model/lstm/zeros_1/mulMul.sequential_1/model/lstm/strided_slice:output:0.sequential_1/model/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/model/lstm/zeros_1/mul?
&sequential_1/model/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_1/model/lstm/zeros_1/Less/y?
$sequential_1/model/lstm/zeros_1/LessLess'sequential_1/model/lstm/zeros_1/mul:z:0/sequential_1/model/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2&
$sequential_1/model/lstm/zeros_1/Less?
(sequential_1/model/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_1/model/lstm/zeros_1/packed/1?
&sequential_1/model/lstm/zeros_1/packedPack.sequential_1/model/lstm/strided_slice:output:01sequential_1/model/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_1/model/lstm/zeros_1/packed?
%sequential_1/model/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%sequential_1/model/lstm/zeros_1/Const?
sequential_1/model/lstm/zeros_1Fill/sequential_1/model/lstm/zeros_1/packed:output:0.sequential_1/model/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2!
sequential_1/model/lstm/zeros_1?
&sequential_1/model/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_1/model/lstm/transpose/perm?
!sequential_1/model/lstm/transpose	Transpose1sequential_1/model/max_pooling1d/Squeeze:output:0/sequential_1/model/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:?????????@2#
!sequential_1/model/lstm/transpose?
sequential_1/model/lstm/Shape_1Shape%sequential_1/model/lstm/transpose:y:0*
T0*
_output_shapes
:2!
sequential_1/model/lstm/Shape_1?
-sequential_1/model/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_1/model/lstm/strided_slice_1/stack?
/sequential_1/model/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_1/model/lstm/strided_slice_1/stack_1?
/sequential_1/model/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_1/model/lstm/strided_slice_1/stack_2?
'sequential_1/model/lstm/strided_slice_1StridedSlice(sequential_1/model/lstm/Shape_1:output:06sequential_1/model/lstm/strided_slice_1/stack:output:08sequential_1/model/lstm/strided_slice_1/stack_1:output:08sequential_1/model/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'sequential_1/model/lstm/strided_slice_1?
3sequential_1/model/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3sequential_1/model/lstm/TensorArrayV2/element_shape?
%sequential_1/model/lstm/TensorArrayV2TensorListReserve<sequential_1/model/lstm/TensorArrayV2/element_shape:output:00sequential_1/model/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_1/model/lstm/TensorArrayV2?
Msequential_1/model/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2O
Msequential_1/model/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape?
?sequential_1/model/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor%sequential_1/model/lstm/transpose:y:0Vsequential_1/model/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?sequential_1/model/lstm/TensorArrayUnstack/TensorListFromTensor?
-sequential_1/model/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_1/model/lstm/strided_slice_2/stack?
/sequential_1/model/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_1/model/lstm/strided_slice_2/stack_1?
/sequential_1/model/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_1/model/lstm/strided_slice_2/stack_2?
'sequential_1/model/lstm/strided_slice_2StridedSlice%sequential_1/model/lstm/transpose:y:06sequential_1/model/lstm/strided_slice_2/stack:output:08sequential_1/model/lstm/strided_slice_2/stack_1:output:08sequential_1/model/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2)
'sequential_1/model/lstm/strided_slice_2?
7sequential_1/model/lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp@sequential_1_model_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype029
7sequential_1/model/lstm/lstm_cell/MatMul/ReadVariableOp?
(sequential_1/model/lstm/lstm_cell/MatMulMatMul0sequential_1/model/lstm/strided_slice_2:output:0?sequential_1/model/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(sequential_1/model/lstm/lstm_cell/MatMul?
9sequential_1/model/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBsequential_1_model_lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02;
9sequential_1/model/lstm/lstm_cell/MatMul_1/ReadVariableOp?
*sequential_1/model/lstm/lstm_cell/MatMul_1MatMul&sequential_1/model/lstm/zeros:output:0Asequential_1/model/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_1/model/lstm/lstm_cell/MatMul_1?
%sequential_1/model/lstm/lstm_cell/addAddV22sequential_1/model/lstm/lstm_cell/MatMul:product:04sequential_1/model/lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2'
%sequential_1/model/lstm/lstm_cell/add?
8sequential_1/model/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAsequential_1_model_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8sequential_1/model/lstm/lstm_cell/BiasAdd/ReadVariableOp?
)sequential_1/model/lstm/lstm_cell/BiasAddBiasAdd)sequential_1/model/lstm/lstm_cell/add:z:0@sequential_1/model/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_1/model/lstm/lstm_cell/BiasAdd?
'sequential_1/model/lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/model/lstm/lstm_cell/Const?
1sequential_1/model/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_1/model/lstm/lstm_cell/split/split_dim?
'sequential_1/model/lstm/lstm_cell/splitSplit:sequential_1/model/lstm/lstm_cell/split/split_dim:output:02sequential_1/model/lstm/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2)
'sequential_1/model/lstm/lstm_cell/split?
)sequential_1/model/lstm/lstm_cell/SigmoidSigmoid0sequential_1/model/lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2+
)sequential_1/model/lstm/lstm_cell/Sigmoid?
+sequential_1/model/lstm/lstm_cell/Sigmoid_1Sigmoid0sequential_1/model/lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2-
+sequential_1/model/lstm/lstm_cell/Sigmoid_1?
%sequential_1/model/lstm/lstm_cell/mulMul/sequential_1/model/lstm/lstm_cell/Sigmoid_1:y:0(sequential_1/model/lstm/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2'
%sequential_1/model/lstm/lstm_cell/mul?
&sequential_1/model/lstm/lstm_cell/ReluRelu0sequential_1/model/lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2(
&sequential_1/model/lstm/lstm_cell/Relu?
'sequential_1/model/lstm/lstm_cell/mul_1Mul-sequential_1/model/lstm/lstm_cell/Sigmoid:y:04sequential_1/model/lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2)
'sequential_1/model/lstm/lstm_cell/mul_1?
'sequential_1/model/lstm/lstm_cell/add_1AddV2)sequential_1/model/lstm/lstm_cell/mul:z:0+sequential_1/model/lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2)
'sequential_1/model/lstm/lstm_cell/add_1?
+sequential_1/model/lstm/lstm_cell/Sigmoid_2Sigmoid0sequential_1/model/lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2-
+sequential_1/model/lstm/lstm_cell/Sigmoid_2?
(sequential_1/model/lstm/lstm_cell/Relu_1Relu+sequential_1/model/lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2*
(sequential_1/model/lstm/lstm_cell/Relu_1?
'sequential_1/model/lstm/lstm_cell/mul_2Mul/sequential_1/model/lstm/lstm_cell/Sigmoid_2:y:06sequential_1/model/lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2)
'sequential_1/model/lstm/lstm_cell/mul_2?
5sequential_1/model/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    27
5sequential_1/model/lstm/TensorArrayV2_1/element_shape?
'sequential_1/model/lstm/TensorArrayV2_1TensorListReserve>sequential_1/model/lstm/TensorArrayV2_1/element_shape:output:00sequential_1/model/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'sequential_1/model/lstm/TensorArrayV2_1~
sequential_1/model/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_1/model/lstm/time?
0sequential_1/model/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_1/model/lstm/while/maximum_iterations?
*sequential_1/model/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/model/lstm/while/loop_counter?
sequential_1/model/lstm/whileWhile3sequential_1/model/lstm/while/loop_counter:output:09sequential_1/model/lstm/while/maximum_iterations:output:0%sequential_1/model/lstm/time:output:00sequential_1/model/lstm/TensorArrayV2_1:handle:0&sequential_1/model/lstm/zeros:output:0(sequential_1/model/lstm/zeros_1:output:00sequential_1/model/lstm/strided_slice_1:output:0Osequential_1/model/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_1_model_lstm_lstm_cell_matmul_readvariableop_resourceBsequential_1_model_lstm_lstm_cell_matmul_1_readvariableop_resourceAsequential_1_model_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*4
body,R*
(sequential_1_model_lstm_while_body_15360*4
cond,R*
(sequential_1_model_lstm_while_cond_15359*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
sequential_1/model/lstm/while?
Hsequential_1/model/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2J
Hsequential_1/model/lstm/TensorArrayV2Stack/TensorListStack/element_shape?
:sequential_1/model/lstm/TensorArrayV2Stack/TensorListStackTensorListStack&sequential_1/model/lstm/while:output:3Qsequential_1/model/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype02<
:sequential_1/model/lstm/TensorArrayV2Stack/TensorListStack?
-sequential_1/model/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2/
-sequential_1/model/lstm/strided_slice_3/stack?
/sequential_1/model/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_1/model/lstm/strided_slice_3/stack_1?
/sequential_1/model/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_1/model/lstm/strided_slice_3/stack_2?
'sequential_1/model/lstm/strided_slice_3StridedSliceCsequential_1/model/lstm/TensorArrayV2Stack/TensorListStack:tensor:06sequential_1/model/lstm/strided_slice_3/stack:output:08sequential_1/model/lstm/strided_slice_3/stack_1:output:08sequential_1/model/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2)
'sequential_1/model/lstm/strided_slice_3?
(sequential_1/model/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(sequential_1/model/lstm/transpose_1/perm?
#sequential_1/model/lstm/transpose_1	TransposeCsequential_1/model/lstm/TensorArrayV2Stack/TensorListStack:tensor:01sequential_1/model/lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2%
#sequential_1/model/lstm/transpose_1?
sequential_1/model/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_1/model/lstm/runtime?
8sequential_1/model/last_hidden_state/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2:
8sequential_1/model/last_hidden_state/strided_slice/stack?
:sequential_1/model/last_hidden_state/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2<
:sequential_1/model/last_hidden_state/strided_slice/stack_1?
:sequential_1/model/last_hidden_state/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2<
:sequential_1/model/last_hidden_state/strided_slice/stack_2?
2sequential_1/model/last_hidden_state/strided_sliceStridedSlice'sequential_1/model/lstm/transpose_1:y:0Asequential_1/model/last_hidden_state/strided_slice/stack:output:0Csequential_1/model/last_hidden_state/strided_slice/stack_1:output:0Csequential_1/model/last_hidden_state/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *

begin_mask*
end_mask*
shrink_axis_mask24
2sequential_1/model/last_hidden_state/strided_slice?
?sequential_1/model/attention_score_vec/Tensordot/ReadVariableOpReadVariableOpHsequential_1_model_attention_score_vec_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?sequential_1/model/attention_score_vec/Tensordot/ReadVariableOp?
5sequential_1/model/attention_score_vec/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/model/attention_score_vec/Tensordot/axes?
5sequential_1/model/attention_score_vec/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5sequential_1/model/attention_score_vec/Tensordot/free?
6sequential_1/model/attention_score_vec/Tensordot/ShapeShape'sequential_1/model/lstm/transpose_1:y:0*
T0*
_output_shapes
:28
6sequential_1/model/attention_score_vec/Tensordot/Shape?
>sequential_1/model/attention_score_vec/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>sequential_1/model/attention_score_vec/Tensordot/GatherV2/axis?
9sequential_1/model/attention_score_vec/Tensordot/GatherV2GatherV2?sequential_1/model/attention_score_vec/Tensordot/Shape:output:0>sequential_1/model/attention_score_vec/Tensordot/free:output:0Gsequential_1/model/attention_score_vec/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9sequential_1/model/attention_score_vec/Tensordot/GatherV2?
@sequential_1/model/attention_score_vec/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@sequential_1/model/attention_score_vec/Tensordot/GatherV2_1/axis?
;sequential_1/model/attention_score_vec/Tensordot/GatherV2_1GatherV2?sequential_1/model/attention_score_vec/Tensordot/Shape:output:0>sequential_1/model/attention_score_vec/Tensordot/axes:output:0Isequential_1/model/attention_score_vec/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;sequential_1/model/attention_score_vec/Tensordot/GatherV2_1?
6sequential_1/model/attention_score_vec/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_1/model/attention_score_vec/Tensordot/Const?
5sequential_1/model/attention_score_vec/Tensordot/ProdProdBsequential_1/model/attention_score_vec/Tensordot/GatherV2:output:0?sequential_1/model/attention_score_vec/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5sequential_1/model/attention_score_vec/Tensordot/Prod?
8sequential_1/model/attention_score_vec/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8sequential_1/model/attention_score_vec/Tensordot/Const_1?
7sequential_1/model/attention_score_vec/Tensordot/Prod_1ProdDsequential_1/model/attention_score_vec/Tensordot/GatherV2_1:output:0Asequential_1/model/attention_score_vec/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7sequential_1/model/attention_score_vec/Tensordot/Prod_1?
<sequential_1/model/attention_score_vec/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<sequential_1/model/attention_score_vec/Tensordot/concat/axis?
7sequential_1/model/attention_score_vec/Tensordot/concatConcatV2>sequential_1/model/attention_score_vec/Tensordot/free:output:0>sequential_1/model/attention_score_vec/Tensordot/axes:output:0Esequential_1/model/attention_score_vec/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7sequential_1/model/attention_score_vec/Tensordot/concat?
6sequential_1/model/attention_score_vec/Tensordot/stackPack>sequential_1/model/attention_score_vec/Tensordot/Prod:output:0@sequential_1/model/attention_score_vec/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6sequential_1/model/attention_score_vec/Tensordot/stack?
:sequential_1/model/attention_score_vec/Tensordot/transpose	Transpose'sequential_1/model/lstm/transpose_1:y:0@sequential_1/model/attention_score_vec/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2<
:sequential_1/model/attention_score_vec/Tensordot/transpose?
8sequential_1/model/attention_score_vec/Tensordot/ReshapeReshape>sequential_1/model/attention_score_vec/Tensordot/transpose:y:0?sequential_1/model/attention_score_vec/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8sequential_1/model/attention_score_vec/Tensordot/Reshape?
7sequential_1/model/attention_score_vec/Tensordot/MatMulMatMulAsequential_1/model/attention_score_vec/Tensordot/Reshape:output:0Gsequential_1/model/attention_score_vec/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7sequential_1/model/attention_score_vec/Tensordot/MatMul?
8sequential_1/model/attention_score_vec/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8sequential_1/model/attention_score_vec/Tensordot/Const_2?
>sequential_1/model/attention_score_vec/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>sequential_1/model/attention_score_vec/Tensordot/concat_1/axis?
9sequential_1/model/attention_score_vec/Tensordot/concat_1ConcatV2Bsequential_1/model/attention_score_vec/Tensordot/GatherV2:output:0Asequential_1/model/attention_score_vec/Tensordot/Const_2:output:0Gsequential_1/model/attention_score_vec/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9sequential_1/model/attention_score_vec/Tensordot/concat_1?
0sequential_1/model/attention_score_vec/TensordotReshapeAsequential_1/model/attention_score_vec/Tensordot/MatMul:product:0Bsequential_1/model/attention_score_vec/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 22
0sequential_1/model/attention_score_vec/Tensordot?
1sequential_1/model/attention_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_1/model/attention_score/ExpandDims/dim?
-sequential_1/model/attention_score/ExpandDims
ExpandDims;sequential_1/model/last_hidden_state/strided_slice:output:0:sequential_1/model/attention_score/ExpandDims/dim:output:0*
T0*+
_output_shapes
:????????? 2/
-sequential_1/model/attention_score/ExpandDims?
)sequential_1/model/attention_score/MatMulBatchMatMulV29sequential_1/model/attention_score_vec/Tensordot:output:06sequential_1/model/attention_score/ExpandDims:output:0*
T0*+
_output_shapes
:?????????2+
)sequential_1/model/attention_score/MatMul?
(sequential_1/model/attention_score/ShapeShape2sequential_1/model/attention_score/MatMul:output:0*
T0*
_output_shapes
:2*
(sequential_1/model/attention_score/Shape?
*sequential_1/model/attention_score/SqueezeSqueeze2sequential_1/model/attention_score/MatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims

?????????2,
*sequential_1/model/attention_score/Squeeze?
+sequential_1/model/attention_weight/SoftmaxSoftmax3sequential_1/model/attention_score/Squeeze:output:0*
T0*'
_output_shapes
:?????????2-
+sequential_1/model/attention_weight/Softmax?
0sequential_1/model/context_vector/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential_1/model/context_vector/ExpandDims/dim?
,sequential_1/model/context_vector/ExpandDims
ExpandDims5sequential_1/model/attention_weight/Softmax:softmax:09sequential_1/model/context_vector/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2.
,sequential_1/model/context_vector/ExpandDims?
0sequential_1/model/context_vector/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          22
0sequential_1/model/context_vector/transpose/perm?
+sequential_1/model/context_vector/transpose	Transpose'sequential_1/model/lstm/transpose_1:y:09sequential_1/model/context_vector/transpose/perm:output:0*
T0*+
_output_shapes
:????????? 2-
+sequential_1/model/context_vector/transpose?
(sequential_1/model/context_vector/MatMulBatchMatMulV2/sequential_1/model/context_vector/transpose:y:05sequential_1/model/context_vector/ExpandDims:output:0*
T0*+
_output_shapes
:????????? 2*
(sequential_1/model/context_vector/MatMul?
'sequential_1/model/context_vector/ShapeShape1sequential_1/model/context_vector/MatMul:output:0*
T0*
_output_shapes
:2)
'sequential_1/model/context_vector/Shape?
)sequential_1/model/context_vector/SqueezeSqueeze1sequential_1/model/context_vector/MatMul:output:0*
T0*'
_output_shapes
:????????? *
squeeze_dims

?????????2+
)sequential_1/model/context_vector/Squeeze?
/sequential_1/model/attention_output/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_1/model/attention_output/concat/axis?
*sequential_1/model/attention_output/concatConcatV22sequential_1/model/context_vector/Squeeze:output:0;sequential_1/model/last_hidden_state/strided_slice:output:08sequential_1/model/attention_output/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2,
*sequential_1/model/attention_output/concat?
9sequential_1/model/attention_vector/MatMul/ReadVariableOpReadVariableOpBsequential_1_model_attention_vector_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02;
9sequential_1/model/attention_vector/MatMul/ReadVariableOp?
*sequential_1/model/attention_vector/MatMulMatMul3sequential_1/model/attention_output/concat:output:0Asequential_1/model/attention_vector/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_1/model/attention_vector/MatMul?
(sequential_1/model/attention_vector/TanhTanh4sequential_1/model/attention_vector/MatMul:product:0*
T0*(
_output_shapes
:??????????2*
(sequential_1/model/attention_vector/Tanh?
#sequential_1/model/dropout/IdentityIdentity,sequential_1/model/attention_vector/Tanh:y:0*
T0*(
_output_shapes
:??????????2%
#sequential_1/model/dropout/Identity?
0sequential_1/model/dense_1/MatMul/ReadVariableOpReadVariableOp9sequential_1_model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype022
0sequential_1/model/dense_1/MatMul/ReadVariableOp?
!sequential_1/model/dense_1/MatMulMatMul,sequential_1/model/dropout/Identity:output:08sequential_1/model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2#
!sequential_1/model/dense_1/MatMul?
1sequential_1/model/dense_1/BiasAdd/ReadVariableOpReadVariableOp:sequential_1_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1sequential_1/model/dense_1/BiasAdd/ReadVariableOp?
"sequential_1/model/dense_1/BiasAddBiasAdd+sequential_1/model/dense_1/MatMul:product:09sequential_1/model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2$
"sequential_1/model/dense_1/BiasAdd?
sequential_1/model/dense_1/ReluRelu+sequential_1/model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2!
sequential_1/model/dense_1/Relu?
0sequential_1/model/dense_2/MatMul/ReadVariableOpReadVariableOp9sequential_1_model_dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype022
0sequential_1/model/dense_2/MatMul/ReadVariableOp?
!sequential_1/model/dense_2/MatMulMatMul-sequential_1/model/dense_1/Relu:activations:08sequential_1/model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2#
!sequential_1/model/dense_2/MatMul?
1sequential_1/model/dense_2/BiasAdd/ReadVariableOpReadVariableOp:sequential_1_model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1sequential_1/model/dense_2/BiasAdd/ReadVariableOp?
"sequential_1/model/dense_2/BiasAddBiasAdd+sequential_1/model/dense_2/MatMul:product:09sequential_1/model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2$
"sequential_1/model/dense_2/BiasAdd?
sequential_1/model/dense_2/ReluRelu+sequential_1/model/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2!
sequential_1/model/dense_2/Relu?
(sequential_1/dense/MatMul/ReadVariableOpReadVariableOp1sequential_1_dense_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02*
(sequential_1/dense/MatMul/ReadVariableOp?
sequential_1/dense/MatMulMatMul-sequential_1/model/dense_2/Relu:activations:00sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_1/dense/MatMul?
)sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)sequential_1/dense/BiasAdd/ReadVariableOp?
sequential_1/dense/BiasAddBiasAdd#sequential_1/dense/MatMul:product:01sequential_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_1/dense/BiasAdd?
sequential_1/dense/ReluRelu#sequential_1/dense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
sequential_1/dense/Relu?
sequential_1/dropout_1/IdentityIdentity%sequential_1/dense/Relu:activations:0*
T0*'
_output_shapes
:????????? 2!
sequential_1/dropout_1/Identity?
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOp?
sequential_1/dense_1/MatMulMatMul(sequential_1/dropout_1/Identity:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_1/dense_1/MatMul?
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOp?
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_1/dense_1/BiasAdd?
sequential_1/dense_1/SoftmaxSoftmax%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_1/dense_1/Softmax?
IdentityIdentity&sequential_1/dense_1/Softmax:softmax:0^sequential_1/model/lstm/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????:::::::::::::::::::2>
sequential_1/model/lstm/whilesequential_1/model/lstm/while:X T
+
_output_shapes
:?????????
%
_user_specified_namemodel_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
?
?
while_cond_16022
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_16022___redundant_placeholder0-
)while_cond_16022___redundant_placeholder1-
)while_cond_16022___redundant_placeholder2-
)while_cond_16022___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
|
'__inference_dense_1_layer_call_fn_18830

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_172212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
\
0__inference_attention_output_layer_call_fn_19603
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_attention_output_layer_call_and_return_conditional_losses_167162
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :????????? :Q M
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?
C
'__inference_dropout_layer_call_fn_19645

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_167622
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_17805

inputs<
8model_conv1d_conv1d_expanddims_1_readvariableop_resource0
,model_conv1d_biasadd_readvariableop_resource>
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource2
.model_conv1d_1_biasadd_readvariableop_resource>
:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource2
.model_conv1d_2_biasadd_readvariableop_resource7
3model_lstm_lstm_cell_matmul_readvariableop_resource9
5model_lstm_lstm_cell_matmul_1_readvariableop_resource8
4model_lstm_lstm_cell_biasadd_readvariableop_resource?
;model_attention_score_vec_tensordot_readvariableop_resource9
5model_attention_vector_matmul_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??model/lstm/while?
model/conv1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
model/conv1d/Pad/paddings?
model/conv1d/PadPadinputs"model/conv1d/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2
model/conv1d/Pad?
"model/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"model/conv1d/conv1d/ExpandDims/dim?
model/conv1d/conv1d/ExpandDims
ExpandDimsmodel/conv1d/Pad:output:0+model/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2 
model/conv1d/conv1d/ExpandDims?
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8model_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype021
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
$model/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$model/conv1d/conv1d/ExpandDims_1/dim?
 model/conv1d/conv1d/ExpandDims_1
ExpandDims7model/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0-model/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2"
 model/conv1d/conv1d/ExpandDims_1?
model/conv1d/conv1dConv2D'model/conv1d/conv1d/ExpandDims:output:0)model/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
model/conv1d/conv1d?
model/conv1d/conv1d/SqueezeSqueezemodel/conv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
model/conv1d/conv1d/Squeeze?
#model/conv1d/BiasAdd/ReadVariableOpReadVariableOp,model_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#model/conv1d/BiasAdd/ReadVariableOp?
model/conv1d/BiasAddBiasAdd$model/conv1d/conv1d/Squeeze:output:0+model/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model/conv1d/BiasAdd?
model/conv1d/ReluRelumodel/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model/conv1d/Relu?
model/conv1d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
model/conv1d_1/Pad/paddings?
model/conv1d_1/PadPadmodel/conv1d/Relu:activations:0$model/conv1d_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model/conv1d_1/Pad?
$model/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$model/conv1d_1/conv1d/ExpandDims/dim?
 model/conv1d_1/conv1d/ExpandDims
ExpandDimsmodel/conv1d_1/Pad:output:0-model/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2"
 model/conv1d_1/conv1d/ExpandDims?
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype023
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
&model/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/conv1d_1/conv1d/ExpandDims_1/dim?
"model/conv1d_1/conv1d/ExpandDims_1
ExpandDims9model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2$
"model/conv1d_1/conv1d/ExpandDims_1?
model/conv1d_1/conv1dConv2D)model/conv1d_1/conv1d/ExpandDims:output:0+model/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
model/conv1d_1/conv1d?
model/conv1d_1/conv1d/SqueezeSqueezemodel/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
model/conv1d_1/conv1d/Squeeze?
%model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv1d_1/BiasAdd/ReadVariableOp?
model/conv1d_1/BiasAddBiasAdd&model/conv1d_1/conv1d/Squeeze:output:0-model/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model/conv1d_1/BiasAdd?
model/conv1d_1/ReluRelumodel/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model/conv1d_1/Relu?
model/conv1d_2/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
model/conv1d_2/Pad/paddings?
model/conv1d_2/PadPad!model/conv1d_1/Relu:activations:0$model/conv1d_2/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model/conv1d_2/Pad?
$model/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$model/conv1d_2/conv1d/ExpandDims/dim?
 model/conv1d_2/conv1d/ExpandDims
ExpandDimsmodel/conv1d_2/Pad:output:0-model/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2"
 model/conv1d_2/conv1d/ExpandDims?
1model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype023
1model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
&model/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/conv1d_2/conv1d/ExpandDims_1/dim?
"model/conv1d_2/conv1d/ExpandDims_1
ExpandDims9model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2$
"model/conv1d_2/conv1d/ExpandDims_1?
model/conv1d_2/conv1dConv2D)model/conv1d_2/conv1d/ExpandDims:output:0+model/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
model/conv1d_2/conv1d?
model/conv1d_2/conv1d/SqueezeSqueezemodel/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
model/conv1d_2/conv1d/Squeeze?
%model/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv1d_2/BiasAdd/ReadVariableOp?
model/conv1d_2/BiasAddBiasAdd&model/conv1d_2/conv1d/Squeeze:output:0-model/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model/conv1d_2/BiasAdd?
model/conv1d_2/ReluRelumodel/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model/conv1d_2/Relu?
"model/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"model/max_pooling1d/ExpandDims/dim?
model/max_pooling1d/ExpandDims
ExpandDims!model/conv1d_2/Relu:activations:0+model/max_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2 
model/max_pooling1d/ExpandDims?
model/max_pooling1d/MaxPoolMaxPool'model/max_pooling1d/ExpandDims:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
model/max_pooling1d/MaxPool?
model/max_pooling1d/SqueezeSqueeze$model/max_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
model/max_pooling1d/Squeezex
model/lstm/ShapeShape$model/max_pooling1d/Squeeze:output:0*
T0*
_output_shapes
:2
model/lstm/Shape?
model/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
model/lstm/strided_slice/stack?
 model/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 model/lstm/strided_slice/stack_1?
 model/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 model/lstm/strided_slice/stack_2?
model/lstm/strided_sliceStridedSlicemodel/lstm/Shape:output:0'model/lstm/strided_slice/stack:output:0)model/lstm/strided_slice/stack_1:output:0)model/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/lstm/strided_slicer
model/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
model/lstm/zeros/mul/y?
model/lstm/zeros/mulMul!model/lstm/strided_slice:output:0model/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model/lstm/zeros/mulu
model/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model/lstm/zeros/Less/y?
model/lstm/zeros/LessLessmodel/lstm/zeros/mul:z:0 model/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model/lstm/zeros/Lessx
model/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
model/lstm/zeros/packed/1?
model/lstm/zeros/packedPack!model/lstm/strided_slice:output:0"model/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model/lstm/zeros/packedu
model/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lstm/zeros/Const?
model/lstm/zerosFill model/lstm/zeros/packed:output:0model/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
model/lstm/zerosv
model/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
model/lstm/zeros_1/mul/y?
model/lstm/zeros_1/mulMul!model/lstm/strided_slice:output:0!model/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model/lstm/zeros_1/muly
model/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model/lstm/zeros_1/Less/y?
model/lstm/zeros_1/LessLessmodel/lstm/zeros_1/mul:z:0"model/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model/lstm/zeros_1/Less|
model/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
model/lstm/zeros_1/packed/1?
model/lstm/zeros_1/packedPack!model/lstm/strided_slice:output:0$model/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
model/lstm/zeros_1/packedy
model/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lstm/zeros_1/Const?
model/lstm/zeros_1Fill"model/lstm/zeros_1/packed:output:0!model/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
model/lstm/zeros_1?
model/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model/lstm/transpose/perm?
model/lstm/transpose	Transpose$model/max_pooling1d/Squeeze:output:0"model/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:?????????@2
model/lstm/transposep
model/lstm/Shape_1Shapemodel/lstm/transpose:y:0*
T0*
_output_shapes
:2
model/lstm/Shape_1?
 model/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 model/lstm/strided_slice_1/stack?
"model/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"model/lstm/strided_slice_1/stack_1?
"model/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"model/lstm/strided_slice_1/stack_2?
model/lstm/strided_slice_1StridedSlicemodel/lstm/Shape_1:output:0)model/lstm/strided_slice_1/stack:output:0+model/lstm/strided_slice_1/stack_1:output:0+model/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/lstm/strided_slice_1?
&model/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&model/lstm/TensorArrayV2/element_shape?
model/lstm/TensorArrayV2TensorListReserve/model/lstm/TensorArrayV2/element_shape:output:0#model/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model/lstm/TensorArrayV2?
@model/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2B
@model/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape?
2model/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/lstm/transpose:y:0Imodel/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2model/lstm/TensorArrayUnstack/TensorListFromTensor?
 model/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 model/lstm/strided_slice_2/stack?
"model/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"model/lstm/strided_slice_2/stack_1?
"model/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"model/lstm/strided_slice_2/stack_2?
model/lstm/strided_slice_2StridedSlicemodel/lstm/transpose:y:0)model/lstm/strided_slice_2/stack:output:0+model/lstm/strided_slice_2/stack_1:output:0+model/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
model/lstm/strided_slice_2?
*model/lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp3model_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02,
*model/lstm/lstm_cell/MatMul/ReadVariableOp?
model/lstm/lstm_cell/MatMulMatMul#model/lstm/strided_slice_2:output:02model/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/lstm/lstm_cell/MatMul?
,model/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp5model_lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02.
,model/lstm/lstm_cell/MatMul_1/ReadVariableOp?
model/lstm/lstm_cell/MatMul_1MatMulmodel/lstm/zeros:output:04model/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/lstm/lstm_cell/MatMul_1?
model/lstm/lstm_cell/addAddV2%model/lstm/lstm_cell/MatMul:product:0'model/lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
model/lstm/lstm_cell/add?
+model/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp4model_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+model/lstm/lstm_cell/BiasAdd/ReadVariableOp?
model/lstm/lstm_cell/BiasAddBiasAddmodel/lstm/lstm_cell/add:z:03model/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/lstm/lstm_cell/BiasAddz
model/lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
model/lstm/lstm_cell/Const?
$model/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$model/lstm/lstm_cell/split/split_dim?
model/lstm/lstm_cell/splitSplit-model/lstm/lstm_cell/split/split_dim:output:0%model/lstm/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
model/lstm/lstm_cell/split?
model/lstm/lstm_cell/SigmoidSigmoid#model/lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
model/lstm/lstm_cell/Sigmoid?
model/lstm/lstm_cell/Sigmoid_1Sigmoid#model/lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2 
model/lstm/lstm_cell/Sigmoid_1?
model/lstm/lstm_cell/mulMul"model/lstm/lstm_cell/Sigmoid_1:y:0model/lstm/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
model/lstm/lstm_cell/mul?
model/lstm/lstm_cell/ReluRelu#model/lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
model/lstm/lstm_cell/Relu?
model/lstm/lstm_cell/mul_1Mul model/lstm/lstm_cell/Sigmoid:y:0'model/lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
model/lstm/lstm_cell/mul_1?
model/lstm/lstm_cell/add_1AddV2model/lstm/lstm_cell/mul:z:0model/lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
model/lstm/lstm_cell/add_1?
model/lstm/lstm_cell/Sigmoid_2Sigmoid#model/lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2 
model/lstm/lstm_cell/Sigmoid_2?
model/lstm/lstm_cell/Relu_1Relumodel/lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
model/lstm/lstm_cell/Relu_1?
model/lstm/lstm_cell/mul_2Mul"model/lstm/lstm_cell/Sigmoid_2:y:0)model/lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
model/lstm/lstm_cell/mul_2?
(model/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2*
(model/lstm/TensorArrayV2_1/element_shape?
model/lstm/TensorArrayV2_1TensorListReserve1model/lstm/TensorArrayV2_1/element_shape:output:0#model/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model/lstm/TensorArrayV2_1d
model/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model/lstm/time?
#model/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#model/lstm/while/maximum_iterations?
model/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
model/lstm/while/loop_counter?
model/lstm/whileWhile&model/lstm/while/loop_counter:output:0,model/lstm/while/maximum_iterations:output:0model/lstm/time:output:0#model/lstm/TensorArrayV2_1:handle:0model/lstm/zeros:output:0model/lstm/zeros_1:output:0#model/lstm/strided_slice_1:output:0Bmodel/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:03model_lstm_lstm_cell_matmul_readvariableop_resource5model_lstm_lstm_cell_matmul_1_readvariableop_resource4model_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*'
bodyR
model_lstm_while_body_17630*'
condR
model_lstm_while_cond_17629*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
model/lstm/while?
;model/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2=
;model/lstm/TensorArrayV2Stack/TensorListStack/element_shape?
-model/lstm/TensorArrayV2Stack/TensorListStackTensorListStackmodel/lstm/while:output:3Dmodel/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype02/
-model/lstm/TensorArrayV2Stack/TensorListStack?
 model/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 model/lstm/strided_slice_3/stack?
"model/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"model/lstm/strided_slice_3/stack_1?
"model/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"model/lstm/strided_slice_3/stack_2?
model/lstm/strided_slice_3StridedSlice6model/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)model/lstm/strided_slice_3/stack:output:0+model/lstm/strided_slice_3/stack_1:output:0+model/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
model/lstm/strided_slice_3?
model/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model/lstm/transpose_1/perm?
model/lstm/transpose_1	Transpose6model/lstm/TensorArrayV2Stack/TensorListStack:tensor:0$model/lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
model/lstm/transpose_1|
model/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lstm/runtime?
+model/last_hidden_state/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2-
+model/last_hidden_state/strided_slice/stack?
-model/last_hidden_state/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2/
-model/last_hidden_state/strided_slice/stack_1?
-model/last_hidden_state/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2/
-model/last_hidden_state/strided_slice/stack_2?
%model/last_hidden_state/strided_sliceStridedSlicemodel/lstm/transpose_1:y:04model/last_hidden_state/strided_slice/stack:output:06model/last_hidden_state/strided_slice/stack_1:output:06model/last_hidden_state/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *

begin_mask*
end_mask*
shrink_axis_mask2'
%model/last_hidden_state/strided_slice?
2model/attention_score_vec/Tensordot/ReadVariableOpReadVariableOp;model_attention_score_vec_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype024
2model/attention_score_vec/Tensordot/ReadVariableOp?
(model/attention_score_vec/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2*
(model/attention_score_vec/Tensordot/axes?
(model/attention_score_vec/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2*
(model/attention_score_vec/Tensordot/free?
)model/attention_score_vec/Tensordot/ShapeShapemodel/lstm/transpose_1:y:0*
T0*
_output_shapes
:2+
)model/attention_score_vec/Tensordot/Shape?
1model/attention_score_vec/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1model/attention_score_vec/Tensordot/GatherV2/axis?
,model/attention_score_vec/Tensordot/GatherV2GatherV22model/attention_score_vec/Tensordot/Shape:output:01model/attention_score_vec/Tensordot/free:output:0:model/attention_score_vec/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,model/attention_score_vec/Tensordot/GatherV2?
3model/attention_score_vec/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3model/attention_score_vec/Tensordot/GatherV2_1/axis?
.model/attention_score_vec/Tensordot/GatherV2_1GatherV22model/attention_score_vec/Tensordot/Shape:output:01model/attention_score_vec/Tensordot/axes:output:0<model/attention_score_vec/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:20
.model/attention_score_vec/Tensordot/GatherV2_1?
)model/attention_score_vec/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)model/attention_score_vec/Tensordot/Const?
(model/attention_score_vec/Tensordot/ProdProd5model/attention_score_vec/Tensordot/GatherV2:output:02model/attention_score_vec/Tensordot/Const:output:0*
T0*
_output_shapes
: 2*
(model/attention_score_vec/Tensordot/Prod?
+model/attention_score_vec/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+model/attention_score_vec/Tensordot/Const_1?
*model/attention_score_vec/Tensordot/Prod_1Prod7model/attention_score_vec/Tensordot/GatherV2_1:output:04model/attention_score_vec/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2,
*model/attention_score_vec/Tensordot/Prod_1?
/model/attention_score_vec/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/model/attention_score_vec/Tensordot/concat/axis?
*model/attention_score_vec/Tensordot/concatConcatV21model/attention_score_vec/Tensordot/free:output:01model/attention_score_vec/Tensordot/axes:output:08model/attention_score_vec/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*model/attention_score_vec/Tensordot/concat?
)model/attention_score_vec/Tensordot/stackPack1model/attention_score_vec/Tensordot/Prod:output:03model/attention_score_vec/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2+
)model/attention_score_vec/Tensordot/stack?
-model/attention_score_vec/Tensordot/transpose	Transposemodel/lstm/transpose_1:y:03model/attention_score_vec/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2/
-model/attention_score_vec/Tensordot/transpose?
+model/attention_score_vec/Tensordot/ReshapeReshape1model/attention_score_vec/Tensordot/transpose:y:02model/attention_score_vec/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2-
+model/attention_score_vec/Tensordot/Reshape?
*model/attention_score_vec/Tensordot/MatMulMatMul4model/attention_score_vec/Tensordot/Reshape:output:0:model/attention_score_vec/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2,
*model/attention_score_vec/Tensordot/MatMul?
+model/attention_score_vec/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2-
+model/attention_score_vec/Tensordot/Const_2?
1model/attention_score_vec/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1model/attention_score_vec/Tensordot/concat_1/axis?
,model/attention_score_vec/Tensordot/concat_1ConcatV25model/attention_score_vec/Tensordot/GatherV2:output:04model/attention_score_vec/Tensordot/Const_2:output:0:model/attention_score_vec/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2.
,model/attention_score_vec/Tensordot/concat_1?
#model/attention_score_vec/TensordotReshape4model/attention_score_vec/Tensordot/MatMul:product:05model/attention_score_vec/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2%
#model/attention_score_vec/Tensordot?
$model/attention_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$model/attention_score/ExpandDims/dim?
 model/attention_score/ExpandDims
ExpandDims.model/last_hidden_state/strided_slice:output:0-model/attention_score/ExpandDims/dim:output:0*
T0*+
_output_shapes
:????????? 2"
 model/attention_score/ExpandDims?
model/attention_score/MatMulBatchMatMulV2,model/attention_score_vec/Tensordot:output:0)model/attention_score/ExpandDims:output:0*
T0*+
_output_shapes
:?????????2
model/attention_score/MatMul?
model/attention_score/ShapeShape%model/attention_score/MatMul:output:0*
T0*
_output_shapes
:2
model/attention_score/Shape?
model/attention_score/SqueezeSqueeze%model/attention_score/MatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims

?????????2
model/attention_score/Squeeze?
model/attention_weight/SoftmaxSoftmax&model/attention_score/Squeeze:output:0*
T0*'
_output_shapes
:?????????2 
model/attention_weight/Softmax?
#model/context_vector/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/context_vector/ExpandDims/dim?
model/context_vector/ExpandDims
ExpandDims(model/attention_weight/Softmax:softmax:0,model/context_vector/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2!
model/context_vector/ExpandDims?
#model/context_vector/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#model/context_vector/transpose/perm?
model/context_vector/transpose	Transposemodel/lstm/transpose_1:y:0,model/context_vector/transpose/perm:output:0*
T0*+
_output_shapes
:????????? 2 
model/context_vector/transpose?
model/context_vector/MatMulBatchMatMulV2"model/context_vector/transpose:y:0(model/context_vector/ExpandDims:output:0*
T0*+
_output_shapes
:????????? 2
model/context_vector/MatMul?
model/context_vector/ShapeShape$model/context_vector/MatMul:output:0*
T0*
_output_shapes
:2
model/context_vector/Shape?
model/context_vector/SqueezeSqueeze$model/context_vector/MatMul:output:0*
T0*'
_output_shapes
:????????? *
squeeze_dims

?????????2
model/context_vector/Squeeze?
"model/attention_output/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"model/attention_output/concat/axis?
model/attention_output/concatConcatV2%model/context_vector/Squeeze:output:0.model/last_hidden_state/strided_slice:output:0+model/attention_output/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
model/attention_output/concat?
,model/attention_vector/MatMul/ReadVariableOpReadVariableOp5model_attention_vector_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02.
,model/attention_vector/MatMul/ReadVariableOp?
model/attention_vector/MatMulMatMul&model/attention_output/concat:output:04model/attention_vector/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/attention_vector/MatMul?
model/attention_vector/TanhTanh'model/attention_vector/MatMul:product:0*
T0*(
_output_shapes
:??????????2
model/attention_vector/Tanh
model/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
model/dropout/dropout/Const?
model/dropout/dropout/MulMulmodel/attention_vector/Tanh:y:0$model/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
model/dropout/dropout/Mul?
model/dropout/dropout/ShapeShapemodel/attention_vector/Tanh:y:0*
T0*
_output_shapes
:2
model/dropout/dropout/Shape?
2model/dropout/dropout/random_uniform/RandomUniformRandomUniform$model/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed?24
2model/dropout/dropout/random_uniform/RandomUniform?
$model/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2&
$model/dropout/dropout/GreaterEqual/y?
"model/dropout/dropout/GreaterEqualGreaterEqual;model/dropout/dropout/random_uniform/RandomUniform:output:0-model/dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"model/dropout/dropout/GreaterEqual?
model/dropout/dropout/CastCast&model/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
model/dropout/dropout/Cast?
model/dropout/dropout/Mul_1Mulmodel/dropout/dropout/Mul:z:0model/dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
model/dropout/dropout/Mul_1?
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02%
#model/dense_1/MatMul/ReadVariableOp?
model/dense_1/MatMulMatMulmodel/dropout/dropout/Mul_1:z:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense_1/MatMul?
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense_1/BiasAdd?
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model/dense_1/Relu?
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#model/dense_2/MatMul/ReadVariableOp?
model/dense_2/MatMulMatMul model/dense_1/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense_2/MatMul?
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp?
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense_2/BiasAdd?
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model/dense_2/Relu?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul model/dense_2/Relu:activations:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2

dense/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const?
dropout_1/dropout/MulMuldense/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_1/dropout/Mulz
dropout_1/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed?*
seed220
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_1/dropout/Mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Softmax?
IdentityIdentitydense_1/Softmax:softmax:0^model/lstm/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????:::::::::::::::::::2$
model/lstm/whilemodel/lstm/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
?
?
A__inference_conv1d_layer_call_and_return_conditional_losses_15540

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingso
PadPadinputsPad/paddings:output:0*
T0*4
_output_shapes"
 :??????????????????2
Padp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????@2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?1
?
while_body_16320
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*lstm_cell_matmul_readvariableop_resource_00
,lstm_cell_matmul_1_readvariableop_resource_0/
+lstm_cell_biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/MatMul/ReadVariableOpReadVariableOp*lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:????????? 2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/mul_2?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identity_4k

Identity_5Identitylstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"X
)lstm_cell_biasadd_readvariableop_resource+lstm_cell_biasadd_readvariableop_resource_0"Z
*lstm_cell_matmul_1_readvariableop_resource,lstm_cell_matmul_1_readvariableop_resource_0"V
(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
`
'__inference_dropout_layer_call_fn_19640

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_167572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19751

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:????????? 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:????????? 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:????????? 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:????????? 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:????????? 2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????@:????????? :????????? ::::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
{
&__inference_conv1d_layer_call_fn_15550

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_155402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_18080

inputs<
8model_conv1d_conv1d_expanddims_1_readvariableop_resource0
,model_conv1d_biasadd_readvariableop_resource>
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource2
.model_conv1d_1_biasadd_readvariableop_resource>
:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource2
.model_conv1d_2_biasadd_readvariableop_resource7
3model_lstm_lstm_cell_matmul_readvariableop_resource9
5model_lstm_lstm_cell_matmul_1_readvariableop_resource8
4model_lstm_lstm_cell_biasadd_readvariableop_resource?
;model_attention_score_vec_tensordot_readvariableop_resource9
5model_attention_vector_matmul_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??model/lstm/while?
model/conv1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
model/conv1d/Pad/paddings?
model/conv1d/PadPadinputs"model/conv1d/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????2
model/conv1d/Pad?
"model/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"model/conv1d/conv1d/ExpandDims/dim?
model/conv1d/conv1d/ExpandDims
ExpandDimsmodel/conv1d/Pad:output:0+model/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2 
model/conv1d/conv1d/ExpandDims?
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8model_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype021
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
$model/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$model/conv1d/conv1d/ExpandDims_1/dim?
 model/conv1d/conv1d/ExpandDims_1
ExpandDims7model/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0-model/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2"
 model/conv1d/conv1d/ExpandDims_1?
model/conv1d/conv1dConv2D'model/conv1d/conv1d/ExpandDims:output:0)model/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
model/conv1d/conv1d?
model/conv1d/conv1d/SqueezeSqueezemodel/conv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
model/conv1d/conv1d/Squeeze?
#model/conv1d/BiasAdd/ReadVariableOpReadVariableOp,model_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#model/conv1d/BiasAdd/ReadVariableOp?
model/conv1d/BiasAddBiasAdd$model/conv1d/conv1d/Squeeze:output:0+model/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model/conv1d/BiasAdd?
model/conv1d/ReluRelumodel/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model/conv1d/Relu?
model/conv1d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
model/conv1d_1/Pad/paddings?
model/conv1d_1/PadPadmodel/conv1d/Relu:activations:0$model/conv1d_1/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model/conv1d_1/Pad?
$model/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$model/conv1d_1/conv1d/ExpandDims/dim?
 model/conv1d_1/conv1d/ExpandDims
ExpandDimsmodel/conv1d_1/Pad:output:0-model/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2"
 model/conv1d_1/conv1d/ExpandDims?
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype023
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
&model/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/conv1d_1/conv1d/ExpandDims_1/dim?
"model/conv1d_1/conv1d/ExpandDims_1
ExpandDims9model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2$
"model/conv1d_1/conv1d/ExpandDims_1?
model/conv1d_1/conv1dConv2D)model/conv1d_1/conv1d/ExpandDims:output:0+model/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
model/conv1d_1/conv1d?
model/conv1d_1/conv1d/SqueezeSqueezemodel/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
model/conv1d_1/conv1d/Squeeze?
%model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv1d_1/BiasAdd/ReadVariableOp?
model/conv1d_1/BiasAddBiasAdd&model/conv1d_1/conv1d/Squeeze:output:0-model/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model/conv1d_1/BiasAdd?
model/conv1d_1/ReluRelumodel/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model/conv1d_1/Relu?
model/conv1d_2/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
model/conv1d_2/Pad/paddings?
model/conv1d_2/PadPad!model/conv1d_1/Relu:activations:0$model/conv1d_2/Pad/paddings:output:0*
T0*+
_output_shapes
:?????????@2
model/conv1d_2/Pad?
$model/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$model/conv1d_2/conv1d/ExpandDims/dim?
 model/conv1d_2/conv1d/ExpandDims
ExpandDimsmodel/conv1d_2/Pad:output:0-model/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2"
 model/conv1d_2/conv1d/ExpandDims?
1model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype023
1model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
&model/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/conv1d_2/conv1d/ExpandDims_1/dim?
"model/conv1d_2/conv1d/ExpandDims_1
ExpandDims9model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2$
"model/conv1d_2/conv1d/ExpandDims_1?
model/conv1d_2/conv1dConv2D)model/conv1d_2/conv1d/ExpandDims:output:0+model/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
model/conv1d_2/conv1d?
model/conv1d_2/conv1d/SqueezeSqueezemodel/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
model/conv1d_2/conv1d/Squeeze?
%model/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv1d_2/BiasAdd/ReadVariableOp?
model/conv1d_2/BiasAddBiasAdd&model/conv1d_2/conv1d/Squeeze:output:0-model/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
model/conv1d_2/BiasAdd?
model/conv1d_2/ReluRelumodel/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
model/conv1d_2/Relu?
"model/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"model/max_pooling1d/ExpandDims/dim?
model/max_pooling1d/ExpandDims
ExpandDims!model/conv1d_2/Relu:activations:0+model/max_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2 
model/max_pooling1d/ExpandDims?
model/max_pooling1d/MaxPoolMaxPool'model/max_pooling1d/ExpandDims:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
model/max_pooling1d/MaxPool?
model/max_pooling1d/SqueezeSqueeze$model/max_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
model/max_pooling1d/Squeezex
model/lstm/ShapeShape$model/max_pooling1d/Squeeze:output:0*
T0*
_output_shapes
:2
model/lstm/Shape?
model/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
model/lstm/strided_slice/stack?
 model/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 model/lstm/strided_slice/stack_1?
 model/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 model/lstm/strided_slice/stack_2?
model/lstm/strided_sliceStridedSlicemodel/lstm/Shape:output:0'model/lstm/strided_slice/stack:output:0)model/lstm/strided_slice/stack_1:output:0)model/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/lstm/strided_slicer
model/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
model/lstm/zeros/mul/y?
model/lstm/zeros/mulMul!model/lstm/strided_slice:output:0model/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model/lstm/zeros/mulu
model/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model/lstm/zeros/Less/y?
model/lstm/zeros/LessLessmodel/lstm/zeros/mul:z:0 model/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model/lstm/zeros/Lessx
model/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
model/lstm/zeros/packed/1?
model/lstm/zeros/packedPack!model/lstm/strided_slice:output:0"model/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model/lstm/zeros/packedu
model/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lstm/zeros/Const?
model/lstm/zerosFill model/lstm/zeros/packed:output:0model/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
model/lstm/zerosv
model/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
model/lstm/zeros_1/mul/y?
model/lstm/zeros_1/mulMul!model/lstm/strided_slice:output:0!model/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model/lstm/zeros_1/muly
model/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model/lstm/zeros_1/Less/y?
model/lstm/zeros_1/LessLessmodel/lstm/zeros_1/mul:z:0"model/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model/lstm/zeros_1/Less|
model/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
model/lstm/zeros_1/packed/1?
model/lstm/zeros_1/packedPack!model/lstm/strided_slice:output:0$model/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
model/lstm/zeros_1/packedy
model/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lstm/zeros_1/Const?
model/lstm/zeros_1Fill"model/lstm/zeros_1/packed:output:0!model/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
model/lstm/zeros_1?
model/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model/lstm/transpose/perm?
model/lstm/transpose	Transpose$model/max_pooling1d/Squeeze:output:0"model/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:?????????@2
model/lstm/transposep
model/lstm/Shape_1Shapemodel/lstm/transpose:y:0*
T0*
_output_shapes
:2
model/lstm/Shape_1?
 model/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 model/lstm/strided_slice_1/stack?
"model/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"model/lstm/strided_slice_1/stack_1?
"model/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"model/lstm/strided_slice_1/stack_2?
model/lstm/strided_slice_1StridedSlicemodel/lstm/Shape_1:output:0)model/lstm/strided_slice_1/stack:output:0+model/lstm/strided_slice_1/stack_1:output:0+model/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/lstm/strided_slice_1?
&model/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&model/lstm/TensorArrayV2/element_shape?
model/lstm/TensorArrayV2TensorListReserve/model/lstm/TensorArrayV2/element_shape:output:0#model/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model/lstm/TensorArrayV2?
@model/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2B
@model/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape?
2model/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/lstm/transpose:y:0Imodel/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2model/lstm/TensorArrayUnstack/TensorListFromTensor?
 model/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 model/lstm/strided_slice_2/stack?
"model/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"model/lstm/strided_slice_2/stack_1?
"model/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"model/lstm/strided_slice_2/stack_2?
model/lstm/strided_slice_2StridedSlicemodel/lstm/transpose:y:0)model/lstm/strided_slice_2/stack:output:0+model/lstm/strided_slice_2/stack_1:output:0+model/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
model/lstm/strided_slice_2?
*model/lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp3model_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02,
*model/lstm/lstm_cell/MatMul/ReadVariableOp?
model/lstm/lstm_cell/MatMulMatMul#model/lstm/strided_slice_2:output:02model/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/lstm/lstm_cell/MatMul?
,model/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp5model_lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02.
,model/lstm/lstm_cell/MatMul_1/ReadVariableOp?
model/lstm/lstm_cell/MatMul_1MatMulmodel/lstm/zeros:output:04model/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/lstm/lstm_cell/MatMul_1?
model/lstm/lstm_cell/addAddV2%model/lstm/lstm_cell/MatMul:product:0'model/lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
model/lstm/lstm_cell/add?
+model/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp4model_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+model/lstm/lstm_cell/BiasAdd/ReadVariableOp?
model/lstm/lstm_cell/BiasAddBiasAddmodel/lstm/lstm_cell/add:z:03model/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/lstm/lstm_cell/BiasAddz
model/lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
model/lstm/lstm_cell/Const?
$model/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$model/lstm/lstm_cell/split/split_dim?
model/lstm/lstm_cell/splitSplit-model/lstm/lstm_cell/split/split_dim:output:0%model/lstm/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
model/lstm/lstm_cell/split?
model/lstm/lstm_cell/SigmoidSigmoid#model/lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:????????? 2
model/lstm/lstm_cell/Sigmoid?
model/lstm/lstm_cell/Sigmoid_1Sigmoid#model/lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:????????? 2 
model/lstm/lstm_cell/Sigmoid_1?
model/lstm/lstm_cell/mulMul"model/lstm/lstm_cell/Sigmoid_1:y:0model/lstm/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
model/lstm/lstm_cell/mul?
model/lstm/lstm_cell/ReluRelu#model/lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
model/lstm/lstm_cell/Relu?
model/lstm/lstm_cell/mul_1Mul model/lstm/lstm_cell/Sigmoid:y:0'model/lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
model/lstm/lstm_cell/mul_1?
model/lstm/lstm_cell/add_1AddV2model/lstm/lstm_cell/mul:z:0model/lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
model/lstm/lstm_cell/add_1?
model/lstm/lstm_cell/Sigmoid_2Sigmoid#model/lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:????????? 2 
model/lstm/lstm_cell/Sigmoid_2?
model/lstm/lstm_cell/Relu_1Relumodel/lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
model/lstm/lstm_cell/Relu_1?
model/lstm/lstm_cell/mul_2Mul"model/lstm/lstm_cell/Sigmoid_2:y:0)model/lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
model/lstm/lstm_cell/mul_2?
(model/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2*
(model/lstm/TensorArrayV2_1/element_shape?
model/lstm/TensorArrayV2_1TensorListReserve1model/lstm/TensorArrayV2_1/element_shape:output:0#model/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model/lstm/TensorArrayV2_1d
model/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model/lstm/time?
#model/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#model/lstm/while/maximum_iterations?
model/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
model/lstm/while/loop_counter?
model/lstm/whileWhile&model/lstm/while/loop_counter:output:0,model/lstm/while/maximum_iterations:output:0model/lstm/time:output:0#model/lstm/TensorArrayV2_1:handle:0model/lstm/zeros:output:0model/lstm/zeros_1:output:0#model/lstm/strided_slice_1:output:0Bmodel/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:03model_lstm_lstm_cell_matmul_readvariableop_resource5model_lstm_lstm_cell_matmul_1_readvariableop_resource4model_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*'
bodyR
model_lstm_while_body_17919*'
condR
model_lstm_while_cond_17918*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
model/lstm/while?
;model/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2=
;model/lstm/TensorArrayV2Stack/TensorListStack/element_shape?
-model/lstm/TensorArrayV2Stack/TensorListStackTensorListStackmodel/lstm/while:output:3Dmodel/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype02/
-model/lstm/TensorArrayV2Stack/TensorListStack?
 model/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 model/lstm/strided_slice_3/stack?
"model/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"model/lstm/strided_slice_3/stack_1?
"model/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"model/lstm/strided_slice_3/stack_2?
model/lstm/strided_slice_3StridedSlice6model/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)model/lstm/strided_slice_3/stack:output:0+model/lstm/strided_slice_3/stack_1:output:0+model/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
model/lstm/strided_slice_3?
model/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model/lstm/transpose_1/perm?
model/lstm/transpose_1	Transpose6model/lstm/TensorArrayV2Stack/TensorListStack:tensor:0$model/lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
model/lstm/transpose_1|
model/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lstm/runtime?
+model/last_hidden_state/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2-
+model/last_hidden_state/strided_slice/stack?
-model/last_hidden_state/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2/
-model/last_hidden_state/strided_slice/stack_1?
-model/last_hidden_state/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2/
-model/last_hidden_state/strided_slice/stack_2?
%model/last_hidden_state/strided_sliceStridedSlicemodel/lstm/transpose_1:y:04model/last_hidden_state/strided_slice/stack:output:06model/last_hidden_state/strided_slice/stack_1:output:06model/last_hidden_state/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *

begin_mask*
end_mask*
shrink_axis_mask2'
%model/last_hidden_state/strided_slice?
2model/attention_score_vec/Tensordot/ReadVariableOpReadVariableOp;model_attention_score_vec_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype024
2model/attention_score_vec/Tensordot/ReadVariableOp?
(model/attention_score_vec/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2*
(model/attention_score_vec/Tensordot/axes?
(model/attention_score_vec/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2*
(model/attention_score_vec/Tensordot/free?
)model/attention_score_vec/Tensordot/ShapeShapemodel/lstm/transpose_1:y:0*
T0*
_output_shapes
:2+
)model/attention_score_vec/Tensordot/Shape?
1model/attention_score_vec/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1model/attention_score_vec/Tensordot/GatherV2/axis?
,model/attention_score_vec/Tensordot/GatherV2GatherV22model/attention_score_vec/Tensordot/Shape:output:01model/attention_score_vec/Tensordot/free:output:0:model/attention_score_vec/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,model/attention_score_vec/Tensordot/GatherV2?
3model/attention_score_vec/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3model/attention_score_vec/Tensordot/GatherV2_1/axis?
.model/attention_score_vec/Tensordot/GatherV2_1GatherV22model/attention_score_vec/Tensordot/Shape:output:01model/attention_score_vec/Tensordot/axes:output:0<model/attention_score_vec/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:20
.model/attention_score_vec/Tensordot/GatherV2_1?
)model/attention_score_vec/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)model/attention_score_vec/Tensordot/Const?
(model/attention_score_vec/Tensordot/ProdProd5model/attention_score_vec/Tensordot/GatherV2:output:02model/attention_score_vec/Tensordot/Const:output:0*
T0*
_output_shapes
: 2*
(model/attention_score_vec/Tensordot/Prod?
+model/attention_score_vec/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+model/attention_score_vec/Tensordot/Const_1?
*model/attention_score_vec/Tensordot/Prod_1Prod7model/attention_score_vec/Tensordot/GatherV2_1:output:04model/attention_score_vec/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2,
*model/attention_score_vec/Tensordot/Prod_1?
/model/attention_score_vec/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/model/attention_score_vec/Tensordot/concat/axis?
*model/attention_score_vec/Tensordot/concatConcatV21model/attention_score_vec/Tensordot/free:output:01model/attention_score_vec/Tensordot/axes:output:08model/attention_score_vec/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*model/attention_score_vec/Tensordot/concat?
)model/attention_score_vec/Tensordot/stackPack1model/attention_score_vec/Tensordot/Prod:output:03model/attention_score_vec/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2+
)model/attention_score_vec/Tensordot/stack?
-model/attention_score_vec/Tensordot/transpose	Transposemodel/lstm/transpose_1:y:03model/attention_score_vec/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2/
-model/attention_score_vec/Tensordot/transpose?
+model/attention_score_vec/Tensordot/ReshapeReshape1model/attention_score_vec/Tensordot/transpose:y:02model/attention_score_vec/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2-
+model/attention_score_vec/Tensordot/Reshape?
*model/attention_score_vec/Tensordot/MatMulMatMul4model/attention_score_vec/Tensordot/Reshape:output:0:model/attention_score_vec/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2,
*model/attention_score_vec/Tensordot/MatMul?
+model/attention_score_vec/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2-
+model/attention_score_vec/Tensordot/Const_2?
1model/attention_score_vec/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1model/attention_score_vec/Tensordot/concat_1/axis?
,model/attention_score_vec/Tensordot/concat_1ConcatV25model/attention_score_vec/Tensordot/GatherV2:output:04model/attention_score_vec/Tensordot/Const_2:output:0:model/attention_score_vec/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2.
,model/attention_score_vec/Tensordot/concat_1?
#model/attention_score_vec/TensordotReshape4model/attention_score_vec/Tensordot/MatMul:product:05model/attention_score_vec/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2%
#model/attention_score_vec/Tensordot?
$model/attention_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$model/attention_score/ExpandDims/dim?
 model/attention_score/ExpandDims
ExpandDims.model/last_hidden_state/strided_slice:output:0-model/attention_score/ExpandDims/dim:output:0*
T0*+
_output_shapes
:????????? 2"
 model/attention_score/ExpandDims?
model/attention_score/MatMulBatchMatMulV2,model/attention_score_vec/Tensordot:output:0)model/attention_score/ExpandDims:output:0*
T0*+
_output_shapes
:?????????2
model/attention_score/MatMul?
model/attention_score/ShapeShape%model/attention_score/MatMul:output:0*
T0*
_output_shapes
:2
model/attention_score/Shape?
model/attention_score/SqueezeSqueeze%model/attention_score/MatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims

?????????2
model/attention_score/Squeeze?
model/attention_weight/SoftmaxSoftmax&model/attention_score/Squeeze:output:0*
T0*'
_output_shapes
:?????????2 
model/attention_weight/Softmax?
#model/context_vector/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/context_vector/ExpandDims/dim?
model/context_vector/ExpandDims
ExpandDims(model/attention_weight/Softmax:softmax:0,model/context_vector/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2!
model/context_vector/ExpandDims?
#model/context_vector/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#model/context_vector/transpose/perm?
model/context_vector/transpose	Transposemodel/lstm/transpose_1:y:0,model/context_vector/transpose/perm:output:0*
T0*+
_output_shapes
:????????? 2 
model/context_vector/transpose?
model/context_vector/MatMulBatchMatMulV2"model/context_vector/transpose:y:0(model/context_vector/ExpandDims:output:0*
T0*+
_output_shapes
:????????? 2
model/context_vector/MatMul?
model/context_vector/ShapeShape$model/context_vector/MatMul:output:0*
T0*
_output_shapes
:2
model/context_vector/Shape?
model/context_vector/SqueezeSqueeze$model/context_vector/MatMul:output:0*
T0*'
_output_shapes
:????????? *
squeeze_dims

?????????2
model/context_vector/Squeeze?
"model/attention_output/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"model/attention_output/concat/axis?
model/attention_output/concatConcatV2%model/context_vector/Squeeze:output:0.model/last_hidden_state/strided_slice:output:0+model/attention_output/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
model/attention_output/concat?
,model/attention_vector/MatMul/ReadVariableOpReadVariableOp5model_attention_vector_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02.
,model/attention_vector/MatMul/ReadVariableOp?
model/attention_vector/MatMulMatMul&model/attention_output/concat:output:04model/attention_vector/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/attention_vector/MatMul?
model/attention_vector/TanhTanh'model/attention_vector/MatMul:product:0*
T0*(
_output_shapes
:??????????2
model/attention_vector/Tanh?
model/dropout/IdentityIdentitymodel/attention_vector/Tanh:y:0*
T0*(
_output_shapes
:??????????2
model/dropout/Identity?
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02%
#model/dense_1/MatMul/ReadVariableOp?
model/dense_1/MatMulMatMulmodel/dropout/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense_1/MatMul?
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense_1/BiasAdd?
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model/dense_1/Relu?
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#model/dense_2/MatMul/ReadVariableOp?
model/dense_2/MatMulMatMul model/dense_1/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense_2/MatMul?
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp?
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense_2/BiasAdd?
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model/dense_2/Relu?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul model/dense_2/Relu:activations:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2

dense/Relu?
dropout_1/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
dropout_1/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Softmax?
IdentityIdentitydense_1/Softmax:softmax:0^model/lstm/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????:::::::::::::::::::2$
model/lstm/whilemodel/lstm/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
?
?
K__inference_attention_vector_layer_call_and_return_conditional_losses_16733

inputs"
matmul_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMulY
TanhTanhMatMul:product:0*
T0*(
_output_shapes
:??????????2
Tanh]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????@::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
,__inference_sequential_1_layer_call_fn_17374
model_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmodel_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_173332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????:::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_namemodel_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
?@
?
@__inference_model_layer_call_and_return_conditional_losses_17015

inputs
conv1d_16969
conv1d_16971
conv1d_1_16974
conv1d_1_16976
conv1d_2_16979
conv1d_2_16981

lstm_16985

lstm_16987

lstm_16989
attention_score_vec_16993
attention_vector_17000
dense_1_17004
dense_1_17006
dense_2_17009
dense_2_17011
identity??+attention_score_vec/StatefulPartitionedCall?(attention_vector/StatefulPartitionedCall?conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?lstm/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_16969conv1d_16971*
Tin
2*
Tout
2*+
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_155402 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_16974conv1d_1_16976*
Tin
2*
Tout
2*+
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_155692"
 conv1d_1/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_16979conv1d_2_16981*
Tin
2*
Tout
2*+
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_155982"
 conv1d_2/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_156172
max_pooling1d/PartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0
lstm_16985
lstm_16987
lstm_16989*
Tin
2*
Tout
2*+
_output_shapes
:????????? *%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_165582
lstm/StatefulPartitionedCall?
!last_hidden_state/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_last_hidden_state_layer_call_and_return_conditional_losses_166042#
!last_hidden_state/PartitionedCall?
+attention_score_vec/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0attention_score_vec_16993*
Tin
2*
Tout
2*+
_output_shapes
:????????? *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_attention_score_vec_layer_call_and_return_conditional_losses_166442-
+attention_score_vec/StatefulPartitionedCall?
attention_score/PartitionedCallPartitionedCall4attention_score_vec/StatefulPartitionedCall:output:0*last_hidden_state/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_attention_score_layer_call_and_return_conditional_losses_166662!
attention_score/PartitionedCall?
 attention_weight/PartitionedCallPartitionedCall(attention_score/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_attention_weight_layer_call_and_return_conditional_losses_166802"
 attention_weight/PartitionedCall?
context_vector/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0)attention_weight/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_context_vector_layer_call_and_return_conditional_losses_167002 
context_vector/PartitionedCall?
 attention_output/PartitionedCallPartitionedCall'context_vector/PartitionedCall:output:0*last_hidden_state/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_attention_output_layer_call_and_return_conditional_losses_167162"
 attention_output/PartitionedCall?
(attention_vector/StatefulPartitionedCallStatefulPartitionedCall)attention_output/PartitionedCall:output:0attention_vector_17000*
Tin
2*
Tout
2*(
_output_shapes
:??????????*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_attention_vector_layer_call_and_return_conditional_losses_167332*
(attention_vector/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall1attention_vector/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_167622
dropout/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_17004dense_1_17006*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_167862!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_17009dense_2_17011*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_168132!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^attention_score_vec/StatefulPartitionedCall)^attention_vector/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????:::::::::::::::2Z
+attention_score_vec/StatefulPartitionedCall+attention_score_vec/StatefulPartitionedCall2T
(attention_vector/StatefulPartitionedCall(attention_vector/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
model_input8
serving_default_model_input:0?????????;
dense_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"??
_tf_keras_sequential??{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_1", "layers": [{"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_input"}, "name": "conv1d_input", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 20]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["conv1d_input", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d", "trainable": false, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": false, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm", "inbound_nodes": [[["max_pooling1d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "attention_score_vec", "trainable": false, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "attention_score_vec", "inbound_nodes": [[["lstm", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "last_hidden_state", "trainable": false, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAABTAAAAcxYAAAB8AGQAZACFAmQBZABkAIUCZgMZAFMAKQJO6f////+p\nACkB2gF4cgIAAAByAgAAAPU/AAAAL1VzZXJzL2FwcGxlL0Rlc2t0b3AvQUnph4/ljJbmr5Tos70v\n5Lqk5piT57O757WxL21vZGVsL21vZGVsLnB52gg8bGFtYmRhPicAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": [32]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "last_hidden_state", "inbound_nodes": [[["lstm", 0, 0, {}]]]}, {"class_name": "Dot", "config": {"name": "attention_score", "trainable": false, "dtype": "float32", "axes": [2, 1], "normalize": false}, "name": "attention_score", "inbound_nodes": [[["attention_score_vec", 0, 0, {}], ["last_hidden_state", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "attention_weight", "trainable": false, "dtype": "float32", "activation": "softmax"}, "name": "attention_weight", "inbound_nodes": [[["attention_score", 0, 0, {}]]]}, {"class_name": "Dot", "config": {"name": "context_vector", "trainable": false, "dtype": "float32", "axes": [1, 1], "normalize": false}, "name": "context_vector", "inbound_nodes": [[["lstm", 0, 0, {}], ["attention_weight", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "attention_output", "trainable": false, "dtype": "float32", "axis": -1}, "name": "attention_output", "inbound_nodes": [[["context_vector", 0, 0, {}], ["last_hidden_state", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "attention_vector", "trainable": false, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "attention_vector", "inbound_nodes": [[["attention_output", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["attention_vector", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": false, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": false, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["conv1d_input", 0, 0]], "output_layers": [["dense_2", 0, 0]], "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 20]}}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 20]}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 20]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_input"}, "name": "conv1d_input", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 20]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["conv1d_input", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d", "trainable": false, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": false, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm", "inbound_nodes": [[["max_pooling1d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "attention_score_vec", "trainable": false, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "attention_score_vec", "inbound_nodes": [[["lstm", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "last_hidden_state", "trainable": false, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAABTAAAAcxYAAAB8AGQAZACFAmQBZABkAIUCZgMZAFMAKQJO6f////+p\nACkB2gF4cgIAAAByAgAAAPU/AAAAL1VzZXJzL2FwcGxlL0Rlc2t0b3AvQUnph4/ljJbmr5Tos70v\n5Lqk5piT57O757WxL21vZGVsL21vZGVsLnB52gg8bGFtYmRhPicAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": [32]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "last_hidden_state", "inbound_nodes": [[["lstm", 0, 0, {}]]]}, {"class_name": "Dot", "config": {"name": "attention_score", "trainable": false, "dtype": "float32", "axes": [2, 1], "normalize": false}, "name": "attention_score", "inbound_nodes": [[["attention_score_vec", 0, 0, {}], ["last_hidden_state", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "attention_weight", "trainable": false, "dtype": "float32", "activation": "softmax"}, "name": "attention_weight", "inbound_nodes": [[["attention_score", 0, 0, {}]]]}, {"class_name": "Dot", "config": {"name": "context_vector", "trainable": false, "dtype": "float32", "axes": [1, 1], "normalize": false}, "name": "context_vector", "inbound_nodes": [[["lstm", 0, 0, {}], ["attention_weight", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "attention_output", "trainable": false, "dtype": "float32", "axis": -1}, "name": "attention_output", "inbound_nodes": [[["context_vector", 0, 0, {}], ["last_hidden_state", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "attention_vector", "trainable": false, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "attention_vector", "inbound_nodes": [[["attention_output", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["attention_vector", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": false, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": false, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["conv1d_input", 0, 0]], "output_layers": [["dense_2", 0, 0]], "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 20]}}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 20]}}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 1.9999999494757503e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?~
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
layer-8
layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?z
_tf_keras_model?y{"class_name": "Model", "name": "model", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_input"}, "name": "conv1d_input", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 20]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["conv1d_input", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d", "trainable": false, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": false, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm", "inbound_nodes": [[["max_pooling1d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "attention_score_vec", "trainable": false, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "attention_score_vec", "inbound_nodes": [[["lstm", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "last_hidden_state", "trainable": false, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAABTAAAAcxYAAAB8AGQAZACFAmQBZABkAIUCZgMZAFMAKQJO6f////+p\nACkB2gF4cgIAAAByAgAAAPU/AAAAL1VzZXJzL2FwcGxlL0Rlc2t0b3AvQUnph4/ljJbmr5Tos70v\n5Lqk5piT57O757WxL21vZGVsL21vZGVsLnB52gg8bGFtYmRhPicAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": [32]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "last_hidden_state", "inbound_nodes": [[["lstm", 0, 0, {}]]]}, {"class_name": "Dot", "config": {"name": "attention_score", "trainable": false, "dtype": "float32", "axes": [2, 1], "normalize": false}, "name": "attention_score", "inbound_nodes": [[["attention_score_vec", 0, 0, {}], ["last_hidden_state", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "attention_weight", "trainable": false, "dtype": "float32", "activation": "softmax"}, "name": "attention_weight", "inbound_nodes": [[["attention_score", 0, 0, {}]]]}, {"class_name": "Dot", "config": {"name": "context_vector", "trainable": false, "dtype": "float32", "axes": [1, 1], "normalize": false}, "name": "context_vector", "inbound_nodes": [[["lstm", 0, 0, {}], ["attention_weight", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "attention_output", "trainable": false, "dtype": "float32", "axis": -1}, "name": "attention_output", "inbound_nodes": [[["context_vector", 0, 0, {}], ["last_hidden_state", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "attention_vector", "trainable": false, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "attention_vector", "inbound_nodes": [[["attention_output", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["attention_vector", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": false, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": false, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["conv1d_input", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 20]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_input"}, "name": "conv1d_input", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 20]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["conv1d_input", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d", "trainable": false, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": false, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm", "inbound_nodes": [[["max_pooling1d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "attention_score_vec", "trainable": false, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "attention_score_vec", "inbound_nodes": [[["lstm", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "last_hidden_state", "trainable": false, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAABTAAAAcxYAAAB8AGQAZACFAmQBZABkAIUCZgMZAFMAKQJO6f////+p\nACkB2gF4cgIAAAByAgAAAPU/AAAAL1VzZXJzL2FwcGxlL0Rlc2t0b3AvQUnph4/ljJbmr5Tos70v\n5Lqk5piT57O757WxL21vZGVsL21vZGVsLnB52gg8bGFtYmRhPicAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": [32]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "last_hidden_state", "inbound_nodes": [[["lstm", 0, 0, {}]]]}, {"class_name": "Dot", "config": {"name": "attention_score", "trainable": false, "dtype": "float32", "axes": [2, 1], "normalize": false}, "name": "attention_score", "inbound_nodes": [[["attention_score_vec", 0, 0, {}], ["last_hidden_state", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "attention_weight", "trainable": false, "dtype": "float32", "activation": "softmax"}, "name": "attention_weight", "inbound_nodes": [[["attention_score", 0, 0, {}]]]}, {"class_name": "Dot", "config": {"name": "context_vector", "trainable": false, "dtype": "float32", "axes": [1, 1], "normalize": false}, "name": "context_vector", "inbound_nodes": [[["lstm", 0, 0, {}], ["attention_weight", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "attention_output", "trainable": false, "dtype": "float32", "axis": -1}, "name": "attention_output", "inbound_nodes": [[["context_vector", 0, 0, {}], ["last_hidden_state", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "attention_vector", "trainable": false, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "attention_vector", "inbound_nodes": [[["attention_output", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["attention_vector", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": false, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": false, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["conv1d_input", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}}
?

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
%regularization_losses
&	variables
'trainable_variables
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
/iter

0beta_1

1beta_2
	2decay
3learning_ratem? m?)m?*m?v? v?)v?*v?"
	optimizer
 "
trackable_list_wrapper
?
40
51
62
73
84
95
:6
;7
<8
=9
>10
?11
@12
A13
B14
15
 16
)17
*18"
trackable_list_wrapper
<
0
 1
)2
*3"
trackable_list_wrapper
?

Clayers
Dnon_trainable_variables
Elayer_metrics
Flayer_regularization_losses
regularization_losses
	variables
Gmetrics
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv1d_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 20]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_input"}}
?


4kernel
5bias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 20]}, "stateful": false, "config": {"name": "conv1d", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 20]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 20]}}
?	

6kernel
7bias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 64]}}
?	

8kernel
9bias
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 64]}}
?
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling1D", "name": "max_pooling1d", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling1d", "trainable": false, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Xcell
Y
state_spec
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "lstm", "trainable": false, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 64]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 64]}}
?

=kernel
^regularization_losses
_	variables
`trainable_variables
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "attention_score_vec", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "attention_score_vec", "trainable": false, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 32]}}
?
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "last_hidden_state", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "last_hidden_state", "trainable": false, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAABTAAAAcxYAAAB8AGQAZACFAmQBZABkAIUCZgMZAFMAKQJO6f////+p\nACkB2gF4cgIAAAByAgAAAPU/AAAAL1VzZXJzL2FwcGxlL0Rlc2t0b3AvQUnph4/ljJbmr5Tos70v\n5Lqk5piT57O757WxL21vZGVsL21vZGVsLnB52gg8bGFtYmRhPicAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": [32]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?
faxes
gregularization_losses
h	variables
itrainable_variables
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dot", "name": "attention_score", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "attention_score", "trainable": false, "dtype": "float32", "axes": [2, 1], "normalize": false}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 2, 32]}, {"class_name": "TensorShape", "items": [null, 32]}]}
?
kregularization_losses
l	variables
mtrainable_variables
n	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "attention_weight", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "attention_weight", "trainable": false, "dtype": "float32", "activation": "softmax"}}
?
oaxes
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dot", "name": "context_vector", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "context_vector", "trainable": false, "dtype": "float32", "axes": [1, 1], "normalize": false}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 2, 32]}, {"class_name": "TensorShape", "items": [null, 2]}]}
?
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "attention_output", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "attention_output", "trainable": false, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32]}, {"class_name": "TensorShape", "items": [null, 32]}]}
?

>kernel
xregularization_losses
y	variables
ztrainable_variables
{	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "attention_vector", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "attention_vector", "trainable": false, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
|regularization_losses
}	variables
~trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

?kernel
@bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": false, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

Akernel
Bbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_2", "trainable": false, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
 "
trackable_list_wrapper
?
40
51
62
73
84
95
:6
;7
<8
=9
>10
?11
@12
A13
B14"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
regularization_losses
	variables
?metrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:@ 2dense/kernel
: 2
dense/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
!regularization_losses
"	variables
?metrics
#trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
%regularization_losses
&	variables
?metrics
'trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
":  2dense_1_1/kernel
:2dense_1_1/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
+regularization_losses
,	variables
?metrics
-trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
#:!@2conv1d/kernel
:@2conv1d/bias
%:#@@2conv1d_1/kernel
:@2conv1d_1/bias
%:#@@2conv1d_2/kernel
:@2conv1d_2/bias
(:&	@?2lstm/lstm_cell/kernel
2:0	 ?2lstm/lstm_cell/recurrent_kernel
": ?2lstm/lstm_cell/bias
,:*  2attention_score_vec/kernel
*:(	@?2attention_vector/kernel
!:	?@2dense_1/kernel
:@2dense_1/bias
 :@@2dense_2/kernel
:@2dense_2/bias
<
0
1
2
3"
trackable_list_wrapper
?
40
51
62
73
84
95
:6
;7
<8
=9
>10
?11
@12
A13
B14"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
Hregularization_losses
I	variables
?metrics
Jtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
Lregularization_losses
M	variables
?metrics
Ntrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
Pregularization_losses
Q	variables
?metrics
Rtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
Tregularization_losses
U	variables
?metrics
Vtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

:kernel
;recurrent_kernel
<bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "lstm_cell", "trainable": false, "dtype": "float32", "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?states
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
Zregularization_losses
[	variables
?metrics
\trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
^regularization_losses
_	variables
?metrics
`trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
bregularization_losses
c	variables
?metrics
dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
gregularization_losses
h	variables
?metrics
itrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
kregularization_losses
l	variables
?metrics
mtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
pregularization_losses
q	variables
?metrics
rtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
tregularization_losses
u	variables
?metrics
vtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
>0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
xregularization_losses
y	variables
?metrics
ztrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
|regularization_losses
}	variables
?metrics
~trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
?
40
51
62
73
84
95
:6
;7
<8
=9
>10
?11
@12
A13
B14"
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
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
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
5
:0
;1
<2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?	variables
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
X0"
trackable_list_wrapper
 "
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
=0"
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
'
>0"
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
.
?0
@1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
 "
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
#:!@ 2Adam/dense/kernel/m
: 2Adam/dense/bias/m
':% 2Adam/dense_1_1/kernel/m
!:2Adam/dense_1_1/bias/m
#:!@ 2Adam/dense/kernel/v
: 2Adam/dense/bias/v
':% 2Adam/dense_1_1/kernel/v
!:2Adam/dense_1_1/bias/v
?2?
,__inference_sequential_1_layer_call_fn_17374
,__inference_sequential_1_layer_call_fn_18166
,__inference_sequential_1_layer_call_fn_18123
,__inference_sequential_1_layer_call_fn_17463?
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
G__inference_sequential_1_layer_call_and_return_conditional_losses_17805
G__inference_sequential_1_layer_call_and_return_conditional_losses_18080
G__inference_sequential_1_layer_call_and_return_conditional_losses_17284
G__inference_sequential_1_layer_call_and_return_conditional_losses_17238?
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
?2?
 __inference__wrapped_model_15521?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
model_input?????????
?2?
%__inference_model_layer_call_fn_18763
%__inference_model_layer_call_fn_17048
%__inference_model_layer_call_fn_18728
%__inference_model_layer_call_fn_16964?
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
@__inference_model_layer_call_and_return_conditional_losses_18433
@__inference_model_layer_call_and_return_conditional_losses_16879
@__inference_model_layer_call_and_return_conditional_losses_18693
@__inference_model_layer_call_and_return_conditional_losses_16830?
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
?2?
%__inference_dense_layer_call_fn_18783?
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
@__inference_dense_layer_call_and_return_conditional_losses_18774?
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
)__inference_dropout_1_layer_call_fn_18805
)__inference_dropout_1_layer_call_fn_18810?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_1_layer_call_and_return_conditional_losses_18800
D__inference_dropout_1_layer_call_and_return_conditional_losses_18795?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_18830?
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
B__inference_dense_1_layer_call_and_return_conditional_losses_18821?
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
6B4
#__inference_signature_wrapper_17516model_input
?2?
&__inference_conv1d_layer_call_fn_15550?
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
annotations? **?'
%?"??????????????????
?2?
A__inference_conv1d_layer_call_and_return_conditional_losses_15540?
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
annotations? **?'
%?"??????????????????
?2?
(__inference_conv1d_1_layer_call_fn_15579?
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
annotations? **?'
%?"??????????????????@
?2?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_15569?
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
annotations? **?'
%?"??????????????????@
?2?
(__inference_conv1d_2_layer_call_fn_15608?
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
annotations? **?'
%?"??????????????????@
?2?
C__inference_conv1d_2_layer_call_and_return_conditional_losses_15598?
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
annotations? **?'
%?"??????????????????@
?2?
-__inference_max_pooling1d_layer_call_fn_15623?
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
annotations? *3?0
.?+'???????????????????????????
?2?
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_15617?
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
annotations? *3?0
.?+'???????????????????????????
?2?
$__inference_lstm_layer_call_fn_19475
$__inference_lstm_layer_call_fn_19486
$__inference_lstm_layer_call_fn_19147
$__inference_lstm_layer_call_fn_19158?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_lstm_layer_call_and_return_conditional_losses_19136
?__inference_lstm_layer_call_and_return_conditional_losses_19464
?__inference_lstm_layer_call_and_return_conditional_losses_18983
?__inference_lstm_layer_call_and_return_conditional_losses_19311?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_attention_score_vec_layer_call_fn_19520?
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
N__inference_attention_score_vec_layer_call_and_return_conditional_losses_19513?
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
1__inference_last_hidden_state_layer_call_fn_19546
1__inference_last_hidden_state_layer_call_fn_19541?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_last_hidden_state_layer_call_and_return_conditional_losses_19536
L__inference_last_hidden_state_layer_call_and_return_conditional_losses_19528?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_attention_score_layer_call_fn_19562?
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
J__inference_attention_score_layer_call_and_return_conditional_losses_19556?
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
0__inference_attention_weight_layer_call_fn_19572?
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
K__inference_attention_weight_layer_call_and_return_conditional_losses_19567?
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
.__inference_context_vector_layer_call_fn_19590?
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
I__inference_context_vector_layer_call_and_return_conditional_losses_19584?
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
0__inference_attention_output_layer_call_fn_19603?
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
K__inference_attention_output_layer_call_and_return_conditional_losses_19597?
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
0__inference_attention_vector_layer_call_fn_19618?
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
K__inference_attention_vector_layer_call_and_return_conditional_losses_19611?
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
'__inference_dropout_layer_call_fn_19640
'__inference_dropout_layer_call_fn_19645?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_19635
B__inference_dropout_layer_call_and_return_conditional_losses_19630?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_19665?
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
B__inference_dense_1_layer_call_and_return_conditional_losses_19656?
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
'__inference_dense_2_layer_call_fn_19685?
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
B__inference_dense_2_layer_call_and_return_conditional_losses_19676?
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
)__inference_lstm_cell_layer_call_fn_19785
)__inference_lstm_cell_layer_call_fn_19768?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19718
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19751?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
 __inference__wrapped_model_15521?456789:;<=>?@AB )*8?5
.?+
)?&
model_input?????????
? "1?.
,
dense_1!?
dense_1??????????
K__inference_attention_output_layer_call_and_return_conditional_losses_19597?Z?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1????????? 
? "%?"
?
0?????????@
? ?
0__inference_attention_output_layer_call_fn_19603vZ?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1????????? 
? "??????????@?
J__inference_attention_score_layer_call_and_return_conditional_losses_19556?^?[
T?Q
O?L
&?#
inputs/0????????? 
"?
inputs/1????????? 
? "%?"
?
0?????????
? ?
/__inference_attention_score_layer_call_fn_19562z^?[
T?Q
O?L
&?#
inputs/0????????? 
"?
inputs/1????????? 
? "???????????
N__inference_attention_score_vec_layer_call_and_return_conditional_losses_19513c=3?0
)?&
$?!
inputs????????? 
? ")?&
?
0????????? 
? ?
3__inference_attention_score_vec_layer_call_fn_19520V=3?0
)?&
$?!
inputs????????? 
? "?????????? ?
K__inference_attention_vector_layer_call_and_return_conditional_losses_19611\>/?,
%?"
 ?
inputs?????????@
? "&?#
?
0??????????
? ?
0__inference_attention_vector_layer_call_fn_19618O>/?,
%?"
 ?
inputs?????????@
? "????????????
K__inference_attention_weight_layer_call_and_return_conditional_losses_19567X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
0__inference_attention_weight_layer_call_fn_19572K/?,
%?"
 ?
inputs?????????
? "???????????
I__inference_context_vector_layer_call_and_return_conditional_losses_19584?^?[
T?Q
O?L
&?#
inputs/0????????? 
"?
inputs/1?????????
? "%?"
?
0????????? 
? ?
.__inference_context_vector_layer_call_fn_19590z^?[
T?Q
O?L
&?#
inputs/0????????? 
"?
inputs/1?????????
? "?????????? ?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_15569v67<?9
2?/
-?*
inputs??????????????????@
? "2?/
(?%
0??????????????????@
? ?
(__inference_conv1d_1_layer_call_fn_15579i67<?9
2?/
-?*
inputs??????????????????@
? "%?"??????????????????@?
C__inference_conv1d_2_layer_call_and_return_conditional_losses_15598v89<?9
2?/
-?*
inputs??????????????????@
? "2?/
(?%
0??????????????????@
? ?
(__inference_conv1d_2_layer_call_fn_15608i89<?9
2?/
-?*
inputs??????????????????@
? "%?"??????????????????@?
A__inference_conv1d_layer_call_and_return_conditional_losses_15540v45<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????@
? ?
&__inference_conv1d_layer_call_fn_15550i45<?9
2?/
-?*
inputs??????????????????
? "%?"??????????????????@?
B__inference_dense_1_layer_call_and_return_conditional_losses_18821\)*/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? ?
B__inference_dense_1_layer_call_and_return_conditional_losses_19656]?@0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? z
'__inference_dense_1_layer_call_fn_18830O)*/?,
%?"
 ?
inputs????????? 
? "??????????{
'__inference_dense_1_layer_call_fn_19665P?@0?-
&?#
!?
inputs??????????
? "??????????@?
B__inference_dense_2_layer_call_and_return_conditional_losses_19676\AB/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? z
'__inference_dense_2_layer_call_fn_19685OAB/?,
%?"
 ?
inputs?????????@
? "??????????@?
@__inference_dense_layer_call_and_return_conditional_losses_18774\ /?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? x
%__inference_dense_layer_call_fn_18783O /?,
%?"
 ?
inputs?????????@
? "?????????? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_18795\3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_18800\3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? |
)__inference_dropout_1_layer_call_fn_18805O3?0
)?&
 ?
inputs????????? 
p
? "?????????? |
)__inference_dropout_1_layer_call_fn_18810O3?0
)?&
 ?
inputs????????? 
p 
? "?????????? ?
B__inference_dropout_layer_call_and_return_conditional_losses_19630^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_19635^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? |
'__inference_dropout_layer_call_fn_19640Q4?1
*?'
!?
inputs??????????
p
? "???????????|
'__inference_dropout_layer_call_fn_19645Q4?1
*?'
!?
inputs??????????
p 
? "????????????
L__inference_last_hidden_state_layer_call_and_return_conditional_losses_19528d;?8
1?.
$?!
inputs????????? 

 
p
? "%?"
?
0????????? 
? ?
L__inference_last_hidden_state_layer_call_and_return_conditional_losses_19536d;?8
1?.
$?!
inputs????????? 

 
p 
? "%?"
?
0????????? 
? ?
1__inference_last_hidden_state_layer_call_fn_19541W;?8
1?.
$?!
inputs????????? 

 
p
? "?????????? ?
1__inference_last_hidden_state_layer_call_fn_19546W;?8
1?.
$?!
inputs????????? 

 
p 
? "?????????? ?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19718?:;<??}
v?s
 ?
inputs?????????@
K?H
"?
states/0????????? 
"?
states/1????????? 
p
? "s?p
i?f
?
0/0????????? 
E?B
?
0/1/0????????? 
?
0/1/1????????? 
? ?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19751?:;<??}
v?s
 ?
inputs?????????@
K?H
"?
states/0????????? 
"?
states/1????????? 
p 
? "s?p
i?f
?
0/0????????? 
E?B
?
0/1/0????????? 
?
0/1/1????????? 
? ?
)__inference_lstm_cell_layer_call_fn_19768?:;<??}
v?s
 ?
inputs?????????@
K?H
"?
states/0????????? 
"?
states/1????????? 
p
? "c?`
?
0????????? 
A?>
?
1/0????????? 
?
1/1????????? ?
)__inference_lstm_cell_layer_call_fn_19785?:;<??}
v?s
 ?
inputs?????????@
K?H
"?
states/0????????? 
"?
states/1????????? 
p 
? "c?`
?
0????????? 
A?>
?
1/0????????? 
?
1/1????????? ?
?__inference_lstm_layer_call_and_return_conditional_losses_18983?:;<O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p

 
? "2?/
(?%
0?????????????????? 
? ?
?__inference_lstm_layer_call_and_return_conditional_losses_19136?:;<O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p 

 
? "2?/
(?%
0?????????????????? 
? ?
?__inference_lstm_layer_call_and_return_conditional_losses_19311q:;<??<
5?2
$?!
inputs?????????@

 
p

 
? ")?&
?
0????????? 
? ?
?__inference_lstm_layer_call_and_return_conditional_losses_19464q:;<??<
5?2
$?!
inputs?????????@

 
p 

 
? ")?&
?
0????????? 
? ?
$__inference_lstm_layer_call_fn_19147}:;<O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p

 
? "%?"?????????????????? ?
$__inference_lstm_layer_call_fn_19158}:;<O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p 

 
? "%?"?????????????????? ?
$__inference_lstm_layer_call_fn_19475d:;<??<
5?2
$?!
inputs?????????@

 
p

 
? "?????????? ?
$__inference_lstm_layer_call_fn_19486d:;<??<
5?2
$?!
inputs?????????@

 
p 

 
? "?????????? ?
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_15617?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
-__inference_max_pooling1d_layer_call_fn_15623wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
@__inference_model_layer_call_and_return_conditional_losses_16830{456789:;<=>?@ABA?>
7?4
*?'
conv1d_input?????????
p

 
? "%?"
?
0?????????@
? ?
@__inference_model_layer_call_and_return_conditional_losses_16879{456789:;<=>?@ABA?>
7?4
*?'
conv1d_input?????????
p 

 
? "%?"
?
0?????????@
? ?
@__inference_model_layer_call_and_return_conditional_losses_18433u456789:;<=>?@AB;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????@
? ?
@__inference_model_layer_call_and_return_conditional_losses_18693u456789:;<=>?@AB;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????@
? ?
%__inference_model_layer_call_fn_16964n456789:;<=>?@ABA?>
7?4
*?'
conv1d_input?????????
p

 
? "??????????@?
%__inference_model_layer_call_fn_17048n456789:;<=>?@ABA?>
7?4
*?'
conv1d_input?????????
p 

 
? "??????????@?
%__inference_model_layer_call_fn_18728h456789:;<=>?@AB;?8
1?.
$?!
inputs?????????
p

 
? "??????????@?
%__inference_model_layer_call_fn_18763h456789:;<=>?@AB;?8
1?.
$?!
inputs?????????
p 

 
? "??????????@?
G__inference_sequential_1_layer_call_and_return_conditional_losses_17238~456789:;<=>?@AB )*@?=
6?3
)?&
model_input?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_17284~456789:;<=>?@AB )*@?=
6?3
)?&
model_input?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_17805y456789:;<=>?@AB )*;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_18080y456789:;<=>?@AB )*;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
,__inference_sequential_1_layer_call_fn_17374q456789:;<=>?@AB )*@?=
6?3
)?&
model_input?????????
p

 
? "???????????
,__inference_sequential_1_layer_call_fn_17463q456789:;<=>?@AB )*@?=
6?3
)?&
model_input?????????
p 

 
? "???????????
,__inference_sequential_1_layer_call_fn_18123l456789:;<=>?@AB )*;?8
1?.
$?!
inputs?????????
p

 
? "???????????
,__inference_sequential_1_layer_call_fn_18166l456789:;<=>?@AB )*;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
#__inference_signature_wrapper_17516?456789:;<=>?@AB )*G?D
? 
=?:
8
model_input)?&
model_input?????????"1?.
,
dense_1!?
dense_1?????????