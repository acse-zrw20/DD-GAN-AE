??
??
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
;
Elu
features"T
activations"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
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
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	?*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
: *
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? 
*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	? 
*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
? 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*? 
value? B?  B? 
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
R
#	variables
$trainable_variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
R
-	variables
.trainable_variables
/regularization_losses
0	keras_api
R
1	variables
2trainable_variables
3regularization_losses
4	keras_api
h

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
8
0
1
2
3
'4
(5
56
67
8
0
1
2
3
'4
(5
56
67
 
?

	variables
trainable_variables

;layers
<layer_regularization_losses
=non_trainable_variables
>metrics
regularization_losses
?layer_metrics
 
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
trainable_variables

@layers
Alayer_regularization_losses
Bnon_trainable_variables
Cmetrics
regularization_losses
Dlayer_metrics
 
 
 
?
	variables
trainable_variables

Elayers
Flayer_regularization_losses
Gnon_trainable_variables
Hmetrics
regularization_losses
Ilayer_metrics
 
 
 
?
	variables
trainable_variables

Jlayers
Klayer_regularization_losses
Lnon_trainable_variables
Mmetrics
regularization_losses
Nlayer_metrics
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
 trainable_variables

Olayers
Player_regularization_losses
Qnon_trainable_variables
Rmetrics
!regularization_losses
Slayer_metrics
 
 
 
?
#	variables
$trainable_variables

Tlayers
Ulayer_regularization_losses
Vnon_trainable_variables
Wmetrics
%regularization_losses
Xlayer_metrics
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
)	variables
*trainable_variables

Ylayers
Zlayer_regularization_losses
[non_trainable_variables
\metrics
+regularization_losses
]layer_metrics
 
 
 
?
-	variables
.trainable_variables

^layers
_layer_regularization_losses
`non_trainable_variables
ametrics
/regularization_losses
blayer_metrics
 
 
 
?
1	variables
2trainable_variables

clayers
dlayer_regularization_losses
enon_trainable_variables
fmetrics
3regularization_losses
glayer_metrics
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61

50
61
 
?
7	variables
8trainable_variables

hlayers
ilayer_regularization_losses
jnon_trainable_variables
kmetrics
9regularization_losses
llayer_metrics
?
0
1
2
3
4
5
6
7
	8
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
?
serving_default_dense_2_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_2_inputdense_2/kerneldense_2/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense_3/kerneldense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_25046320
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpConst*
Tin
2
*
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
!__inference__traced_save_25046942
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense_3/kerneldense_3/bias*
Tin
2	*
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
GPU2*0J 8? *-
f(R&
$__inference__traced_restore_25046976??
?
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_25046758

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_25046818L
9dense_2_kernel_regularizer_square_readvariableop_resource:	?
identity??0dense_2/kernel/Regularizer/Square/ReadVariableOp?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_2_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	?*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
IdentityIdentity"dense_2/kernel/Regularizer/mul:z:01^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp
?
?
E__inference_dense_2_layer_call_and_return_conditional_losses_25045680

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_25045762

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?/conv2d_3/bias/Regularizer/Square/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddo
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Elu?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_3/kernel/Regularizer/Square?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/Const?
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOp?
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_3/bias/Regularizer/Square?
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Const?
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/Sum?
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_3/bias/Regularizer/mul/x?
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mul?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_3/bias/Regularizer/Square/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_3/bias/Regularizer/Square/ReadVariableOp/conv2d_3/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
E__inference_dense_2_layer_call_and_return_conditional_losses_25046603

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_25046840T
:conv2d_2_kernel_regularizer_square_readvariableop_resource:  
identity??1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_2_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_2/kernel/Regularizer/Square?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/Const?
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
IdentityIdentity#conv2d_2/kernel/Regularizer/mul:z:02^conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp
?t
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_25046172
dense_2_input#
dense_2_25046098:	?
dense_2_25046100:	?+
conv2d_2_25046105:  
conv2d_2_25046107: +
conv2d_3_25046111: @
conv2d_3_25046113:@#
dense_3_25046118:	? 

dense_3_25046120:

identity?? conv2d_2/StatefulPartitionedCall?/conv2d_2/bias/Regularizer/Square/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp? conv2d_3/StatefulPartitionedCall?/conv2d_3/bias/Regularizer/Square/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_25046098dense_2_25046100*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_250456802!
dense_2/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_250456912
dropout_1/PartitionedCall?
reshape_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_250457072
reshape_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_2_25046105conv2d_2_25046107*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_250457322"
 conv2d_2/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_250456262
up_sampling2d/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_25046111conv2d_3_25046113*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_250457622"
 conv2d_3/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_250456452!
up_sampling2d_1/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall(up_sampling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_250457812
flatten_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_25046118dense_3_25046120*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_250458062!
dense_3/StatefulPartitionedCall?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_25046098*
_output_shapes
:	?*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_25046100*
_output_shapes	
:?*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_25046105*&
_output_shapes
:  *
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_2/kernel/Regularizer/Square?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/Const?
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_25046107*
_output_shapes
: *
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOp?
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_2/bias/Regularizer/Square?
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Const?
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/Sum?
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_2/bias/Regularizer/mul/x?
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mul?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_25046111*&
_output_shapes
: @*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_3/kernel/Regularizer/Square?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/Const?
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_25046113*
_output_shapes
:@*
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOp?
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_3/bias/Regularizer/Square?
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Const?
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/Sum?
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_3/bias/Regularizer/mul/x?
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mul?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_25046118*
_output_shapes
:	? 
*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 
2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_25046120*
_output_shapes
:
*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall0^conv2d_2/bias/Regularizer/Square/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall0^conv2d_3/bias/Regularizer/Square/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2b
/conv2d_2/bias/Regularizer/Square/ReadVariableOp/conv2d_2/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2b
/conv2d_3/bias/Regularizer/Square/ReadVariableOp/conv2d_3/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_2_input
?
N
2__inference_up_sampling2d_1_layer_call_fn_25045651

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
GPU2*0J 8? *V
fQRO
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_250456452
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
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
__inference_loss_fn_1_25046829F
7dense_2_bias_regularizer_square_readvariableop_resource:	?
identity??.dense_2/bias/Regularizer/Square/ReadVariableOp?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_2_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:?*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
IdentityIdentity dense_2/bias/Regularizer/mul:z:0/^dense_2/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_7_25046895E
7dense_3_bias_regularizer_square_readvariableop_resource:

identity??.dense_3/bias/Regularizer/Square/ReadVariableOp?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_3_bias_regularizer_square_readvariableop_resource*
_output_shapes
:
*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
IdentityIdentity dense_3/bias/Regularizer/mul:z:0/^dense_3/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp
??
?	
J__inference_sequential_1_layer_call_and_return_conditional_losses_25046420

inputs9
&dense_2_matmul_readvariableop_resource:	?6
'dense_2_biasadd_readvariableop_resource:	?A
'conv2d_2_conv2d_readvariableop_resource:  6
(conv2d_2_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource: @6
(conv2d_3_biasadd_readvariableop_resource:@9
&dense_3_matmul_readvariableop_resource:	? 
5
'dense_3_biasadd_readvariableop_resource:

identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?/conv2d_2/bias/Regularizer/Square/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?/conv2d_3/bias/Regularizer/Square/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAdd?
dropout_1/IdentityIdentitydense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/Identitym
reshape_1/ShapeShapedropout_1/Identity:output:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapedropout_1/Identity:output:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
reshape_1/Reshape?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dreshape_1/Reshape:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/BiasAddx
conv2d_2/EluEluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/Elu{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1?
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_2/Elu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/BiasAddx
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/Elu
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1?
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Elu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbors
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshape=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0flatten_1/Const:output:0*
T0*(
_output_shapes
:?????????? 2
flatten_1/Reshape?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	? 
*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/BiasAddp
dense_3/TanhTanhdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_3/Tanh?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_2/kernel/Regularizer/Square?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/Const?
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOp?
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_2/bias/Regularizer/Square?
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Const?
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/Sum?
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_2/bias/Regularizer/mul/x?
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mul?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_3/kernel/Regularizer/Square?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/Const?
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOp?
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_3/bias/Regularizer/Square?
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Const?
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/Sum?
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_3/bias/Regularizer/mul/x?
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mul?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	? 
*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 
2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
IdentityIdentitydense_3/Tanh:y:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp0^conv2d_2/bias/Regularizer/Square/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp0^conv2d_3/bias/Regularizer/Square/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2b
/conv2d_2/bias/Regularizer/Square/ReadVariableOp/conv2d_2/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2b
/conv2d_3/bias/Regularizer/Square/ReadVariableOp/conv2d_3/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_4_25046862T
:conv2d_3_kernel_regularizer_square_readvariableop_resource: @
identity??1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_3_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_3/kernel/Regularizer/Square?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/Const?
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
IdentityIdentity#conv2d_3/kernel/Regularizer/mul:z:02^conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp
?	
?
/__inference_sequential_1_layer_call_fn_25046095
dense_2_input
unknown:	?
	unknown_0:	?#
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:	? 

	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_250460552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_2_input
?
?
!__inference__traced_save_25046942
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
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
value	B :2

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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*j
_input_shapesY
W: :	?:?:  : : @:@:	? 
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	? 
: 

_output_shapes
:
:	

_output_shapes
: 
?&
?
$__inference__traced_restore_25046976
file_prefix2
assignvariableop_dense_2_kernel:	?.
assignvariableop_1_dense_2_bias:	?<
"assignvariableop_2_conv2d_2_kernel:  .
 assignvariableop_3_conv2d_2_bias: <
"assignvariableop_4_conv2d_3_kernel: @.
 assignvariableop_5_conv2d_3_bias:@4
!assignvariableop_6_dense_3_kernel:	? 
-
assignvariableop_7_dense_3_bias:


identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
E__inference_dense_3_layer_call_and_return_conditional_losses_25045806

inputs1
matmul_readvariableop_resource:	? 
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? 
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Tanh?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? 
*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 
2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
e
,__inference_dropout_1_layer_call_fn_25046639

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_250459422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_3_layer_call_fn_25046746

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_250457622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?t
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_25045861

inputs#
dense_2_25045681:	?
dense_2_25045683:	?+
conv2d_2_25045733:  
conv2d_2_25045735: +
conv2d_3_25045763: @
conv2d_3_25045765:@#
dense_3_25045807:	? 

dense_3_25045809:

identity?? conv2d_2/StatefulPartitionedCall?/conv2d_2/bias/Regularizer/Square/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp? conv2d_3/StatefulPartitionedCall?/conv2d_3/bias/Regularizer/Square/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_25045681dense_2_25045683*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_250456802!
dense_2/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_250456912
dropout_1/PartitionedCall?
reshape_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_250457072
reshape_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_2_25045733conv2d_2_25045735*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_250457322"
 conv2d_2/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_250456262
up_sampling2d/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_25045763conv2d_3_25045765*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_250457622"
 conv2d_3/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_250456452!
up_sampling2d_1/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall(up_sampling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_250457812
flatten_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_25045807dense_3_25045809*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_250458062!
dense_3/StatefulPartitionedCall?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_25045681*
_output_shapes
:	?*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_25045683*
_output_shapes	
:?*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_25045733*&
_output_shapes
:  *
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_2/kernel/Regularizer/Square?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/Const?
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_25045735*
_output_shapes
: *
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOp?
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_2/bias/Regularizer/Square?
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Const?
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/Sum?
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_2/bias/Regularizer/mul/x?
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mul?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_25045763*&
_output_shapes
: @*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_3/kernel/Regularizer/Square?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/Const?
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_25045765*
_output_shapes
:@*
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOp?
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_3/bias/Regularizer/Square?
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Const?
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/Sum?
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_3/bias/Regularizer/mul/x?
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mul?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_25045807*
_output_shapes
:	? 
*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 
2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_25045809*
_output_shapes
:
*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall0^conv2d_2/bias/Regularizer/Square/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall0^conv2d_3/bias/Regularizer/Square/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2b
/conv2d_2/bias/Regularizer/Square/ReadVariableOp/conv2d_2/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2b
/conv2d_3/bias/Regularizer/Square/ReadVariableOp/conv2d_3/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_3_layer_call_fn_25046807

inputs
unknown:	? 

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_250458062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?v
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_25046055

inputs#
dense_2_25045981:	?
dense_2_25045983:	?+
conv2d_2_25045988:  
conv2d_2_25045990: +
conv2d_3_25045994: @
conv2d_3_25045996:@#
dense_3_25046001:	? 

dense_3_25046003:

identity?? conv2d_2/StatefulPartitionedCall?/conv2d_2/bias/Regularizer/Square/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp? conv2d_3/StatefulPartitionedCall?/conv2d_3/bias/Regularizer/Square/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?!dropout_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_25045981dense_2_25045983*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_250456802!
dense_2/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_250459422#
!dropout_1/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_250457072
reshape_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_2_25045988conv2d_2_25045990*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_250457322"
 conv2d_2/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_250456262
up_sampling2d/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_25045994conv2d_3_25045996*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_250457622"
 conv2d_3/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_250456452!
up_sampling2d_1/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall(up_sampling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_250457812
flatten_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_25046001dense_3_25046003*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_250458062!
dense_3/StatefulPartitionedCall?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_25045981*
_output_shapes
:	?*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_25045983*
_output_shapes	
:?*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_25045988*&
_output_shapes
:  *
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_2/kernel/Regularizer/Square?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/Const?
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_25045990*
_output_shapes
: *
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOp?
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_2/bias/Regularizer/Square?
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Const?
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/Sum?
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_2/bias/Regularizer/mul/x?
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mul?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_25045994*&
_output_shapes
: @*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_3/kernel/Regularizer/Square?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/Const?
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_25045996*
_output_shapes
:@*
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOp?
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_3/bias/Regularizer/Square?
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Const?
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/Sum?
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_3/bias/Regularizer/mul/x?
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mul?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_25046001*
_output_shapes
:	? 
*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 
2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_25046003*
_output_shapes
:
*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall0^conv2d_2/bias/Regularizer/Square/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall0^conv2d_3/bias/Regularizer/Square/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2b
/conv2d_2/bias/Regularizer/Square/ReadVariableOp/conv2d_2/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2b
/conv2d_3/bias/Regularizer/Square/ReadVariableOp/conv2d_3/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
/__inference_sequential_1_layer_call_fn_25046548

inputs
unknown:	?
	unknown_0:	?#
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:	? 

	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_250458612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?S
?
#__inference__wrapped_model_25045613
dense_2_inputF
3sequential_1_dense_2_matmul_readvariableop_resource:	?C
4sequential_1_dense_2_biasadd_readvariableop_resource:	?N
4sequential_1_conv2d_2_conv2d_readvariableop_resource:  C
5sequential_1_conv2d_2_biasadd_readvariableop_resource: N
4sequential_1_conv2d_3_conv2d_readvariableop_resource: @C
5sequential_1_conv2d_3_biasadd_readvariableop_resource:@F
3sequential_1_dense_3_matmul_readvariableop_resource:	? 
B
4sequential_1_dense_3_biasadd_readvariableop_resource:

identity??,sequential_1/conv2d_2/BiasAdd/ReadVariableOp?+sequential_1/conv2d_2/Conv2D/ReadVariableOp?,sequential_1/conv2d_3/BiasAdd/ReadVariableOp?+sequential_1/conv2d_3/Conv2D/ReadVariableOp?+sequential_1/dense_2/BiasAdd/ReadVariableOp?*sequential_1/dense_2/MatMul/ReadVariableOp?+sequential_1/dense_3/BiasAdd/ReadVariableOp?*sequential_1/dense_3/MatMul/ReadVariableOp?
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_1/dense_2/MatMul/ReadVariableOp?
sequential_1/dense_2/MatMulMatMuldense_2_input2sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_2/MatMul?
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOp?
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_2/BiasAdd?
sequential_1/dropout_1/IdentityIdentity%sequential_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2!
sequential_1/dropout_1/Identity?
sequential_1/reshape_1/ShapeShape(sequential_1/dropout_1/Identity:output:0*
T0*
_output_shapes
:2
sequential_1/reshape_1/Shape?
*sequential_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_1/reshape_1/strided_slice/stack?
,sequential_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_1/reshape_1/strided_slice/stack_1?
,sequential_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_1/reshape_1/strided_slice/stack_2?
$sequential_1/reshape_1/strided_sliceStridedSlice%sequential_1/reshape_1/Shape:output:03sequential_1/reshape_1/strided_slice/stack:output:05sequential_1/reshape_1/strided_slice/stack_1:output:05sequential_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_1/reshape_1/strided_slice?
&sequential_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_1/reshape_1/Reshape/shape/1?
&sequential_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_1/reshape_1/Reshape/shape/2?
&sequential_1/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_1/reshape_1/Reshape/shape/3?
$sequential_1/reshape_1/Reshape/shapePack-sequential_1/reshape_1/strided_slice:output:0/sequential_1/reshape_1/Reshape/shape/1:output:0/sequential_1/reshape_1/Reshape/shape/2:output:0/sequential_1/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/reshape_1/Reshape/shape?
sequential_1/reshape_1/ReshapeReshape(sequential_1/dropout_1/Identity:output:0-sequential_1/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2 
sequential_1/reshape_1/Reshape?
+sequential_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+sequential_1/conv2d_2/Conv2D/ReadVariableOp?
sequential_1/conv2d_2/Conv2DConv2D'sequential_1/reshape_1/Reshape:output:03sequential_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential_1/conv2d_2/Conv2D?
,sequential_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/conv2d_2/BiasAdd/ReadVariableOp?
sequential_1/conv2d_2/BiasAddBiasAdd%sequential_1/conv2d_2/Conv2D:output:04sequential_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential_1/conv2d_2/BiasAdd?
sequential_1/conv2d_2/EluElu&sequential_1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_1/conv2d_2/Elu?
 sequential_1/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2"
 sequential_1/up_sampling2d/Const?
"sequential_1/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_1/up_sampling2d/Const_1?
sequential_1/up_sampling2d/mulMul)sequential_1/up_sampling2d/Const:output:0+sequential_1/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2 
sequential_1/up_sampling2d/mul?
7sequential_1/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor'sequential_1/conv2d_2/Elu:activations:0"sequential_1/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(29
7sequential_1/up_sampling2d/resize/ResizeNearestNeighbor?
+sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+sequential_1/conv2d_3/Conv2D/ReadVariableOp?
sequential_1/conv2d_3/Conv2DConv2DHsequential_1/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:03sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential_1/conv2d_3/Conv2D?
,sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp?
sequential_1/conv2d_3/BiasAddBiasAdd%sequential_1/conv2d_3/Conv2D:output:04sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential_1/conv2d_3/BiasAdd?
sequential_1/conv2d_3/EluElu&sequential_1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_1/conv2d_3/Elu?
"sequential_1/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_1/up_sampling2d_1/Const?
$sequential_1/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2&
$sequential_1/up_sampling2d_1/Const_1?
 sequential_1/up_sampling2d_1/mulMul+sequential_1/up_sampling2d_1/Const:output:0-sequential_1/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2"
 sequential_1/up_sampling2d_1/mul?
9sequential_1/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor'sequential_1/conv2d_3/Elu:activations:0$sequential_1/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2;
9sequential_1/up_sampling2d_1/resize/ResizeNearestNeighbor?
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_1/flatten_1/Const?
sequential_1/flatten_1/ReshapeReshapeJsequential_1/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:?????????? 2 
sequential_1/flatten_1/Reshape?
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	? 
*
dtype02,
*sequential_1/dense_3/MatMul/ReadVariableOp?
sequential_1/dense_3/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_1/dense_3/MatMul?
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOp?
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_1/dense_3/BiasAdd?
sequential_1/dense_3/TanhTanh%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
sequential_1/dense_3/Tanh?
IdentityIdentitysequential_1/dense_3/Tanh:y:0-^sequential_1/conv2d_2/BiasAdd/ReadVariableOp,^sequential_1/conv2d_2/Conv2D/ReadVariableOp-^sequential_1/conv2d_3/BiasAdd/ReadVariableOp,^sequential_1/conv2d_3/Conv2D/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2\
,sequential_1/conv2d_2/BiasAdd/ReadVariableOp,sequential_1/conv2d_2/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_2/Conv2D/ReadVariableOp+sequential_1/conv2d_2/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp,sequential_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_3/Conv2D/ReadVariableOp+sequential_1/conv2d_3/Conv2D/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_2_input
?
g
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_25045626

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
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
F__inference_conv2d_3_layer_call_and_return_conditional_losses_25046737

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?/conv2d_3/bias/Regularizer/Square/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddo
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Elu?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_3/kernel/Regularizer/Square?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/Const?
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOp?
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_3/bias/Regularizer/Square?
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Const?
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/Sum?
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_3/bias/Regularizer/mul/x?
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mul?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_3/bias/Regularizer/Square/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_3/bias/Regularizer/Square/ReadVariableOp/conv2d_3/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
&__inference_signature_wrapper_25046320
dense_2_input
unknown:	?
	unknown_0:	?#
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:	? 

	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_250456132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_2_input
?
f
G__inference_dropout_1_layer_call_and_return_conditional_losses_25045942

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
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
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
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
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_25046617

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

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
i
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_25045645

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_flatten_1_layer_call_fn_25046763

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_250457812
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_25045732

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?/conv2d_2/bias/Regularizer/Square/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Elu?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_2/kernel/Regularizer/Square?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/Const?
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOp?
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_2/bias/Regularizer/Square?
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Const?
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/Sum?
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_2/bias/Regularizer/mul/x?
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mul?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_2/bias/Regularizer/Square/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_2/bias/Regularizer/Square/ReadVariableOp/conv2d_2/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
f
G__inference_dropout_1_layer_call_and_return_conditional_losses_25046629

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
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
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
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
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?v
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_25046249
dense_2_input#
dense_2_25046175:	?
dense_2_25046177:	?+
conv2d_2_25046182:  
conv2d_2_25046184: +
conv2d_3_25046188: @
conv2d_3_25046190:@#
dense_3_25046195:	? 

dense_3_25046197:

identity?? conv2d_2/StatefulPartitionedCall?/conv2d_2/bias/Regularizer/Square/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp? conv2d_3/StatefulPartitionedCall?/conv2d_3/bias/Regularizer/Square/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?!dropout_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_25046175dense_2_25046177*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_250456802!
dense_2/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_250459422#
!dropout_1/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_250457072
reshape_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_2_25046182conv2d_2_25046184*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_250457322"
 conv2d_2/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_250456262
up_sampling2d/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_25046188conv2d_3_25046190*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_250457622"
 conv2d_3/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_250456452!
up_sampling2d_1/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall(up_sampling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_250457812
flatten_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_25046195dense_3_25046197*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_250458062!
dense_3/StatefulPartitionedCall?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_25046175*
_output_shapes
:	?*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_25046177*
_output_shapes	
:?*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_25046182*&
_output_shapes
:  *
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_2/kernel/Regularizer/Square?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/Const?
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_25046184*
_output_shapes
: *
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOp?
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_2/bias/Regularizer/Square?
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Const?
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/Sum?
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_2/bias/Regularizer/mul/x?
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mul?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_25046188*&
_output_shapes
: @*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_3/kernel/Regularizer/Square?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/Const?
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_25046190*
_output_shapes
:@*
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOp?
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_3/bias/Regularizer/Square?
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Const?
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/Sum?
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_3/bias/Regularizer/mul/x?
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mul?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_25046195*
_output_shapes
:	? 
*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 
2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_25046197*
_output_shapes
:
*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall0^conv2d_2/bias/Regularizer/Square/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall0^conv2d_3/bias/Regularizer/Square/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2b
/conv2d_2/bias/Regularizer/Square/ReadVariableOp/conv2d_2/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2b
/conv2d_3/bias/Regularizer/Square/ReadVariableOp/conv2d_3/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_2_input
?
c
G__inference_reshape_1_layer_call_and_return_conditional_losses_25046653

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:????????? 2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_25046693

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?/conv2d_2/bias/Regularizer/Square/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Elu?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_2/kernel/Regularizer/Square?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/Const?
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOp?
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_2/bias/Regularizer/Square?
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Const?
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/Sum?
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_2/bias/Regularizer/mul/x?
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mul?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_2/bias/Regularizer/Square/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_2/bias/Regularizer/Square/ReadVariableOp/conv2d_2/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_25046851F
8conv2d_2_bias_regularizer_square_readvariableop_resource: 
identity??/conv2d_2/bias/Regularizer/Square/ReadVariableOp?
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_2_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOp?
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_2/bias/Regularizer/Square?
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Const?
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/Sum?
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_2/bias/Regularizer/mul/x?
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mul?
IdentityIdentity!conv2d_2/bias/Regularizer/mul:z:00^conv2d_2/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv2d_2/bias/Regularizer/Square/ReadVariableOp/conv2d_2/bias/Regularizer/Square/ReadVariableOp
?
H
,__inference_reshape_1_layer_call_fn_25046658

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
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_250457072
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_25045691

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

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
H
,__inference_dropout_1_layer_call_fn_25046634

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_250456912
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_5_25046873F
8conv2d_3_bias_regularizer_square_readvariableop_resource:@
identity??/conv2d_3/bias/Regularizer/Square/ReadVariableOp?
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_3_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOp?
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_3/bias/Regularizer/Square?
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Const?
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/Sum?
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_3/bias/Regularizer/mul/x?
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mul?
IdentityIdentity!conv2d_3/bias/Regularizer/mul:z:00^conv2d_3/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv2d_3/bias/Regularizer/Square/ReadVariableOp/conv2d_3/bias/Regularizer/Square/ReadVariableOp
?
c
G__inference_reshape_1_layer_call_and_return_conditional_losses_25045707

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:????????? 2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?	
J__inference_sequential_1_layer_call_and_return_conditional_losses_25046527

inputs9
&dense_2_matmul_readvariableop_resource:	?6
'dense_2_biasadd_readvariableop_resource:	?A
'conv2d_2_conv2d_readvariableop_resource:  6
(conv2d_2_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource: @6
(conv2d_3_biasadd_readvariableop_resource:@9
&dense_3_matmul_readvariableop_resource:	? 
5
'dense_3_biasadd_readvariableop_resource:

identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?/conv2d_2/bias/Regularizer/Square/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?/conv2d_3/bias/Regularizer/Square/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_1/dropout/Const?
dropout_1/dropout/MulMuldense_2/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/dropout/Mulz
dropout_1/dropout/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_1/dropout/Mul_1m
reshape_1/ShapeShapedropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapedropout_1/dropout/Mul_1:z:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
reshape_1/Reshape?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dreshape_1/Reshape:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/BiasAddx
conv2d_2/EluEluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/Elu{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1?
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_2/Elu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/BiasAddx
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/Elu
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1?
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Elu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbors
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshape=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0flatten_1/Const:output:0*
T0*(
_output_shapes
:?????????? 2
flatten_1/Reshape?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	? 
*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_3/BiasAddp
dense_3/TanhTanhdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_3/Tanh?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2$
"conv2d_2/kernel/Regularizer/Square?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/Const?
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/conv2d_2/bias/Regularizer/Square/ReadVariableOp?
 conv2d_2/bias/Regularizer/SquareSquare7conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2"
 conv2d_2/bias/Regularizer/Square?
conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_2/bias/Regularizer/Const?
conv2d_2/bias/Regularizer/SumSum$conv2d_2/bias/Regularizer/Square:y:0(conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/Sum?
conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_2/bias/Regularizer/mul/x?
conv2d_2/bias/Regularizer/mulMul(conv2d_2/bias/Regularizer/mul/x:output:0&conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_2/bias/Regularizer/mul?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_3/kernel/Regularizer/Square?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_3/kernel/Regularizer/Const?
conv2d_3/kernel/Regularizer/SumSum&conv2d_3/kernel/Regularizer/Square:y:0*conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/conv2d_3/bias/Regularizer/Square/ReadVariableOp?
 conv2d_3/bias/Regularizer/SquareSquare7conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2"
 conv2d_3/bias/Regularizer/Square?
conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_3/bias/Regularizer/Const?
conv2d_3/bias/Regularizer/SumSum$conv2d_3/bias/Regularizer/Square:y:0(conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/Sum?
conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72!
conv2d_3/bias/Regularizer/mul/x?
conv2d_3/bias/Regularizer/mulMul(conv2d_3/bias/Regularizer/mul/x:output:0&conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_3/bias/Regularizer/mul?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	? 
*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 
2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
IdentityIdentitydense_3/Tanh:y:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp0^conv2d_2/bias/Regularizer/Square/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp0^conv2d_3/bias/Regularizer/Square/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2b
/conv2d_2/bias/Regularizer/Square/ReadVariableOp/conv2d_2/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2b
/conv2d_3/bias/Regularizer/Square/ReadVariableOp/conv2d_3/bias/Regularizer/Square/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_3_layer_call_and_return_conditional_losses_25046798

inputs1
matmul_readvariableop_resource:	? 
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? 
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Tanh?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? 
*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 
2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_2_layer_call_fn_25046702

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_250457322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
__inference_loss_fn_6_25046884L
9dense_3_kernel_regularizer_square_readvariableop_resource:	? 

identity??0dense_3/kernel/Regularizer/Square/ReadVariableOp?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_3_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	? 
*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 
2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentity"dense_3/kernel/Regularizer/mul:z:01^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp
?	
?
/__inference_sequential_1_layer_call_fn_25045880
dense_2_input
unknown:	?
	unknown_0:	?#
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:	? 

	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_250458612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_2_input
?
?
*__inference_dense_2_layer_call_fn_25046612

inputs
unknown:	?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_250456802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
/__inference_sequential_1_layer_call_fn_25046569

inputs
unknown:	?
	unknown_0:	?#
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:	? 

	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_250460552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_25045781

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
L
0__inference_up_sampling2d_layer_call_fn_25045632

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
GPU2*0J 8? *T
fORM
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_250456262
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
dense_2_input6
serving_default_dense_2_input:0?????????;
dense_30
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?P
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api

signatures
m_default_save_signature
*n&call_and_return_all_conditional_losses
o__call__"?M
_tf_keras_sequential?L{"name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_2_input"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30]}, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 2, 32]}}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 30]}, "float32", "dense_2_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_2_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30]}, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 2}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 3}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "shared_object_id": 5}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 2, 32]}}, "shared_object_id": 6}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 7}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 8}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "shared_object_id": 10}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 11}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 12}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "shared_object_id": 14}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 15}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 16}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 17}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18}]}}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*p&call_and_return_all_conditional_losses
q__call__"?

_tf_keras_layer?	{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30]}, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 2}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 3}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
?
	variables
trainable_variables
regularization_losses
	keras_api
*r&call_and_return_all_conditional_losses
s__call__"?
_tf_keras_layer?{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "shared_object_id": 5}
?
	variables
trainable_variables
regularization_losses
	keras_api
*t&call_and_return_all_conditional_losses
u__call__"?
_tf_keras_layer?{"name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 2, 32]}}, "shared_object_id": 6}
?

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
*v&call_and_return_all_conditional_losses
w__call__"?
_tf_keras_layer?
{"name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 7}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 8}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 2, 32]}}
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
*x&call_and_return_all_conditional_losses
y__call__"?
_tf_keras_layer?{"name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 22}}
?

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
*z&call_and_return_all_conditional_losses
{__call__"?
_tf_keras_layer?
{"name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 11}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 12}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 32]}}
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
*|&call_and_return_all_conditional_losses
}__call__"?
_tf_keras_layer?{"name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 24}}
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
*~&call_and_return_all_conditional_losses
__call__"?
_tf_keras_layer?{"name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 25}}
?


5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 16}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-06}, "shared_object_id": 17}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4096}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4096]}}
X
0
1
2
3
'4
(5
56
67"
trackable_list_wrapper
X
0
1
2
3
'4
(5
56
67"
trackable_list_wrapper
`
?0
?1
?2
?3
?4
?5
?6
?7"
trackable_list_wrapper
?

	variables
trainable_variables

;layers
<layer_regularization_losses
=non_trainable_variables
>metrics
regularization_losses
?layer_metrics
o__call__
m_default_save_signature
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
!:	?2dense_2/kernel
:?2dense_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
	variables
trainable_variables

@layers
Alayer_regularization_losses
Bnon_trainable_variables
Cmetrics
regularization_losses
Dlayer_metrics
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables

Elayers
Flayer_regularization_losses
Gnon_trainable_variables
Hmetrics
regularization_losses
Ilayer_metrics
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables

Jlayers
Klayer_regularization_losses
Lnon_trainable_variables
Mmetrics
regularization_losses
Nlayer_metrics
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_2/kernel
: 2conv2d_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
	variables
 trainable_variables

Olayers
Player_regularization_losses
Qnon_trainable_variables
Rmetrics
!regularization_losses
Slayer_metrics
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
#	variables
$trainable_variables

Tlayers
Ulayer_regularization_losses
Vnon_trainable_variables
Wmetrics
%regularization_losses
Xlayer_metrics
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_3/kernel
:@2conv2d_3/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
)	variables
*trainable_variables

Ylayers
Zlayer_regularization_losses
[non_trainable_variables
\metrics
+regularization_losses
]layer_metrics
{__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
-	variables
.trainable_variables

^layers
_layer_regularization_losses
`non_trainable_variables
ametrics
/regularization_losses
blayer_metrics
}__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
1	variables
2trainable_variables

clayers
dlayer_regularization_losses
enon_trainable_variables
fmetrics
3regularization_losses
glayer_metrics
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
!:	? 
2dense_3/kernel
:
2dense_3/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
7	variables
8trainable_variables

hlayers
ilayer_regularization_losses
jnon_trainable_variables
kmetrics
9regularization_losses
llayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
_
0
1
2
3
4
5
6
7
	8"
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
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
#__inference__wrapped_model_25045613?
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
annotations? *,?)
'?$
dense_2_input?????????
?2?
J__inference_sequential_1_layer_call_and_return_conditional_losses_25046420
J__inference_sequential_1_layer_call_and_return_conditional_losses_25046527
J__inference_sequential_1_layer_call_and_return_conditional_losses_25046172
J__inference_sequential_1_layer_call_and_return_conditional_losses_25046249?
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
/__inference_sequential_1_layer_call_fn_25045880
/__inference_sequential_1_layer_call_fn_25046548
/__inference_sequential_1_layer_call_fn_25046569
/__inference_sequential_1_layer_call_fn_25046095?
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
E__inference_dense_2_layer_call_and_return_conditional_losses_25046603?
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
*__inference_dense_2_layer_call_fn_25046612?
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
G__inference_dropout_1_layer_call_and_return_conditional_losses_25046617
G__inference_dropout_1_layer_call_and_return_conditional_losses_25046629?
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
,__inference_dropout_1_layer_call_fn_25046634
,__inference_dropout_1_layer_call_fn_25046639?
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
G__inference_reshape_1_layer_call_and_return_conditional_losses_25046653?
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
,__inference_reshape_1_layer_call_fn_25046658?
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
F__inference_conv2d_2_layer_call_and_return_conditional_losses_25046693?
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
+__inference_conv2d_2_layer_call_fn_25046702?
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
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_25045626?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
0__inference_up_sampling2d_layer_call_fn_25045632?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_25046737?
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
+__inference_conv2d_3_layer_call_fn_25046746?
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
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_25045645?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
2__inference_up_sampling2d_1_layer_call_fn_25045651?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_flatten_1_layer_call_and_return_conditional_losses_25046758?
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
,__inference_flatten_1_layer_call_fn_25046763?
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
E__inference_dense_3_layer_call_and_return_conditional_losses_25046798?
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
*__inference_dense_3_layer_call_fn_25046807?
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
__inference_loss_fn_0_25046818?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_25046829?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_25046840?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_25046851?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_4_25046862?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_5_25046873?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_6_25046884?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_7_25046895?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
&__inference_signature_wrapper_25046320dense_2_input"?
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
#__inference__wrapped_model_25045613u'(566?3
,?)
'?$
dense_2_input?????????
? "1?.
,
dense_3!?
dense_3?????????
?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_25046693l7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
+__inference_conv2d_2_layer_call_fn_25046702_7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_25046737?'(I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
+__inference_conv2d_3_layer_call_fn_25046746?'(I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
E__inference_dense_2_layer_call_and_return_conditional_losses_25046603]/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? ~
*__inference_dense_2_layer_call_fn_25046612P/?,
%?"
 ?
inputs?????????
? "????????????
E__inference_dense_3_layer_call_and_return_conditional_losses_25046798e568?5
.?+
)?&
inputs??????????????????
? "%?"
?
0?????????

? ?
*__inference_dense_3_layer_call_fn_25046807X568?5
.?+
)?&
inputs??????????????????
? "??????????
?
G__inference_dropout_1_layer_call_and_return_conditional_losses_25046617^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
G__inference_dropout_1_layer_call_and_return_conditional_losses_25046629^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
,__inference_dropout_1_layer_call_fn_25046634Q4?1
*?'
!?
inputs??????????
p 
? "????????????
,__inference_dropout_1_layer_call_fn_25046639Q4?1
*?'
!?
inputs??????????
p
? "????????????
G__inference_flatten_1_layer_call_and_return_conditional_losses_25046758{I?F
??<
:?7
inputs+???????????????????????????@
? ".?+
$?!
0??????????????????
? ?
,__inference_flatten_1_layer_call_fn_25046763nI?F
??<
:?7
inputs+???????????????????????????@
? "!???????????????????=
__inference_loss_fn_0_25046818?

? 
? "? =
__inference_loss_fn_1_25046829?

? 
? "? =
__inference_loss_fn_2_25046840?

? 
? "? =
__inference_loss_fn_3_25046851?

? 
? "? =
__inference_loss_fn_4_25046862'?

? 
? "? =
__inference_loss_fn_5_25046873(?

? 
? "? =
__inference_loss_fn_6_250468845?

? 
? "? =
__inference_loss_fn_7_250468956?

? 
? "? ?
G__inference_reshape_1_layer_call_and_return_conditional_losses_25046653a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0????????? 
? ?
,__inference_reshape_1_layer_call_fn_25046658T0?-
&?#
!?
inputs??????????
? " ?????????? ?
J__inference_sequential_1_layer_call_and_return_conditional_losses_25046172q'(56>?;
4?1
'?$
dense_2_input?????????
p 

 
? "%?"
?
0?????????

? ?
J__inference_sequential_1_layer_call_and_return_conditional_losses_25046249q'(56>?;
4?1
'?$
dense_2_input?????????
p

 
? "%?"
?
0?????????

? ?
J__inference_sequential_1_layer_call_and_return_conditional_losses_25046420j'(567?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
J__inference_sequential_1_layer_call_and_return_conditional_losses_25046527j'(567?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????

? ?
/__inference_sequential_1_layer_call_fn_25045880d'(56>?;
4?1
'?$
dense_2_input?????????
p 

 
? "??????????
?
/__inference_sequential_1_layer_call_fn_25046095d'(56>?;
4?1
'?$
dense_2_input?????????
p

 
? "??????????
?
/__inference_sequential_1_layer_call_fn_25046548]'(567?4
-?*
 ?
inputs?????????
p 

 
? "??????????
?
/__inference_sequential_1_layer_call_fn_25046569]'(567?4
-?*
 ?
inputs?????????
p

 
? "??????????
?
&__inference_signature_wrapper_25046320?'(56G?D
? 
=?:
8
dense_2_input'?$
dense_2_input?????????"1?.
,
dense_3!?
dense_3?????????
?
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_25045645?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_up_sampling2d_1_layer_call_fn_25045651?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_25045626?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_up_sampling2d_layer_call_fn_25045632?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????