??
??
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
?
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

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
?
	MaxPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
?
conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d/kernel
{
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel**
_output_shapes
: *
dtype0
n
conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
: *
dtype0
?
conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv3d_1/kernel

#conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_1/kernel**
_output_shapes
: @*
dtype0
r
conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d_1/bias
k
!conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv3d_1/bias*
_output_shapes
:@*
dtype0
?
conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@?* 
shared_nameconv3d_2/kernel
?
#conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv3d_2/kernel*+
_output_shapes
:@?*
dtype0
s
conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3d_2/bias
l
!conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv3d_2/bias*
_output_shapes	
:?*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?H
*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?H
*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
?!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*? 
value? B?  B? 
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	
signatures
#
_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
?

kernel
bias
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
w
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
?

kernel
bias
#_self_saveable_object_factories
trainable_variables
regularization_losses
 	variables
!	keras_api
w
#"_self_saveable_object_factories
#trainable_variables
$regularization_losses
%	variables
&	keras_api
?

'kernel
(bias
#)_self_saveable_object_factories
*trainable_variables
+regularization_losses
,	variables
-	keras_api
w
#._self_saveable_object_factories
/trainable_variables
0regularization_losses
1	variables
2	keras_api
w
#3_self_saveable_object_factories
4trainable_variables
5regularization_losses
6	variables
7	keras_api
?

8kernel
9bias
#:_self_saveable_object_factories
;trainable_variables
<regularization_losses
=	variables
>	keras_api
 
 
8
0
1
2
3
'4
(5
86
97
 
8
0
1
2
3
'4
(5
86
97
?
trainable_variables
regularization_losses
?non_trainable_variables
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables

Clayers
YW
VARIABLE_VALUEconv3d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv3d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

0
1
?
trainable_variables
regularization_losses
Dnon_trainable_variables
Elayer_regularization_losses
Fmetrics
Glayer_metrics
	variables

Hlayers
 
 
 
 
?
trainable_variables
regularization_losses
Inon_trainable_variables
Jlayer_regularization_losses
Kmetrics
Llayer_metrics
	variables

Mlayers
[Y
VARIABLE_VALUEconv3d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

0
1
?
trainable_variables
regularization_losses
Nnon_trainable_variables
Olayer_regularization_losses
Pmetrics
Qlayer_metrics
 	variables

Rlayers
 
 
 
 
?
#trainable_variables
$regularization_losses
Snon_trainable_variables
Tlayer_regularization_losses
Umetrics
Vlayer_metrics
%	variables

Wlayers
[Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1
 

'0
(1
?
*trainable_variables
+regularization_losses
Xnon_trainable_variables
Ylayer_regularization_losses
Zmetrics
[layer_metrics
,	variables

\layers
 
 
 
 
?
/trainable_variables
0regularization_losses
]non_trainable_variables
^layer_regularization_losses
_metrics
`layer_metrics
1	variables

alayers
 
 
 
 
?
4trainable_variables
5regularization_losses
bnon_trainable_variables
clayer_regularization_losses
dmetrics
elayer_metrics
6	variables

flayers
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

80
91
 

80
91
?
;trainable_variables
<regularization_losses
gnon_trainable_variables
hlayer_regularization_losses
imetrics
jlayer_metrics
=	variables

klayers
 
 
 
 
8
0
1
2
3
4
5
6
7
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
serving_default_conv3d_inputPlaceholder*3
_output_shapes!
:?????????<*
dtype0*(
shape:?????????<
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv3d_inputconv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_2/kernelconv3d_2/biasdense/kernel
dense/bias*
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
GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_423697
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv3d/kernel/Read/ReadVariableOpconv3d/bias/Read/ReadVariableOp#conv3d_1/kernel/Read/ReadVariableOp!conv3d_1/bias/Read/ReadVariableOp#conv3d_2/kernel/Read/ReadVariableOp!conv3d_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpConst*
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
GPU2*0J 8? *(
f#R!
__inference__traced_save_423948
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_2/kernelconv3d_2/biasdense/kernel
dense/bias*
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
GPU2*0J 8? *+
f&R$
"__inference__traced_restore_423982??
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_423443

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? $  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????H2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling3d_1_layer_call_fn_423364

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_4233582
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_layer_call_and_return_conditional_losses_423892

inputs1
matmul_readvariableop_resource:	?H
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?H
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
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????H
 
_user_specified_nameinputs
?
?
&__inference_dense_layer_call_fn_423901

inputs
unknown:	?H
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
GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4234552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????H: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????H
 
_user_specified_nameinputs
?/
?
F__inference_sequential_layer_call_and_return_conditional_losses_423733

inputsC
%conv3d_conv3d_readvariableop_resource: 4
&conv3d_biasadd_readvariableop_resource: E
'conv3d_1_conv3d_readvariableop_resource: @6
(conv3d_1_biasadd_readvariableop_resource:@F
'conv3d_2_conv3d_readvariableop_resource:@?7
(conv3d_2_biasadd_readvariableop_resource:	?7
$dense_matmul_readvariableop_resource:	?H
3
%dense_biasadd_readvariableop_resource:

identity??conv3d/BiasAdd/ReadVariableOp?conv3d/Conv3D/ReadVariableOp?conv3d_1/BiasAdd/ReadVariableOp?conv3d_1/Conv3D/ReadVariableOp?conv3d_2/BiasAdd/ReadVariableOp?conv3d_2/Conv3D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02
conv3d/Conv3D/ReadVariableOp?
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????< *
paddingSAME*
strides	
2
conv3d/Conv3D?
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3d/BiasAdd/ReadVariableOp?
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????< 2
conv3d/BiasAddy
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????< 2
conv3d/Relu?
max_pooling3d/MaxPool3D	MaxPool3Dconv3d/Relu:activations:0*
T0*3
_output_shapes!
:?????????

 *
ksize	
*
paddingSAME*
strides	
2
max_pooling3d/MaxPool3D?
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02 
conv3d_1/Conv3D/ReadVariableOp?
conv3d_1/Conv3DConv3D max_pooling3d/MaxPool3D:output:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

@*
paddingSAME*
strides	
2
conv3d_1/Conv3D?
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_1/BiasAdd/ReadVariableOp?
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

@2
conv3d_1/BiasAdd
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????

@2
conv3d_1/Relu?
max_pooling3d_1/MaxPool3D	MaxPool3Dconv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:?????????@*
ksize	
*
paddingSAME*
strides	
2
max_pooling3d_1/MaxPool3D?
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02 
conv3d_2/Conv3D/ReadVariableOp?
conv3d_2/Conv3DConv3D"max_pooling3d_1/MaxPool3D:output:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
2
conv3d_2/Conv3D?
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv3d_2/BiasAdd/ReadVariableOp?
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????2
conv3d_2/BiasAdd?
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
conv3d_2/Relu?
max_pooling3d_2/MaxPool3D	MaxPool3Dconv3d_2/Relu:activations:0*
T0*4
_output_shapes"
 :??????????*
ksize	
*
paddingSAME*
strides	
2
max_pooling3d_2/MaxPool3Do
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? $  2
flatten/Const?
flatten/ReshapeReshape"max_pooling3d_2/MaxPool3D:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????H2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?H
*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense/BiasAdd?
IdentityIdentitydense/BiasAdd:output:0^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????<: : : : : : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:?????????<
 
_user_specified_nameinputs
?
g
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_423358

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingSAME*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?:
?
!__inference__wrapped_model_423340
conv3d_inputN
0sequential_conv3d_conv3d_readvariableop_resource: ?
1sequential_conv3d_biasadd_readvariableop_resource: P
2sequential_conv3d_1_conv3d_readvariableop_resource: @A
3sequential_conv3d_1_biasadd_readvariableop_resource:@Q
2sequential_conv3d_2_conv3d_readvariableop_resource:@?B
3sequential_conv3d_2_biasadd_readvariableop_resource:	?B
/sequential_dense_matmul_readvariableop_resource:	?H
>
0sequential_dense_biasadd_readvariableop_resource:

identity??(sequential/conv3d/BiasAdd/ReadVariableOp?'sequential/conv3d/Conv3D/ReadVariableOp?*sequential/conv3d_1/BiasAdd/ReadVariableOp?)sequential/conv3d_1/Conv3D/ReadVariableOp?*sequential/conv3d_2/BiasAdd/ReadVariableOp?)sequential/conv3d_2/Conv3D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv3d/Conv3D/ReadVariableOpReadVariableOp0sequential_conv3d_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02)
'sequential/conv3d/Conv3D/ReadVariableOp?
sequential/conv3d/Conv3DConv3Dconv3d_input/sequential/conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????< *
paddingSAME*
strides	
2
sequential/conv3d/Conv3D?
(sequential/conv3d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv3d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv3d/BiasAdd/ReadVariableOp?
sequential/conv3d/BiasAddBiasAdd!sequential/conv3d/Conv3D:output:00sequential/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????< 2
sequential/conv3d/BiasAdd?
sequential/conv3d/ReluRelu"sequential/conv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????< 2
sequential/conv3d/Relu?
"sequential/max_pooling3d/MaxPool3D	MaxPool3D$sequential/conv3d/Relu:activations:0*
T0*3
_output_shapes!
:?????????

 *
ksize	
*
paddingSAME*
strides	
2$
"sequential/max_pooling3d/MaxPool3D?
)sequential/conv3d_1/Conv3D/ReadVariableOpReadVariableOp2sequential_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02+
)sequential/conv3d_1/Conv3D/ReadVariableOp?
sequential/conv3d_1/Conv3DConv3D+sequential/max_pooling3d/MaxPool3D:output:01sequential/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

@*
paddingSAME*
strides	
2
sequential/conv3d_1/Conv3D?
*sequential/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv3d_1/BiasAdd/ReadVariableOp?
sequential/conv3d_1/BiasAddBiasAdd#sequential/conv3d_1/Conv3D:output:02sequential/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

@2
sequential/conv3d_1/BiasAdd?
sequential/conv3d_1/ReluRelu$sequential/conv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????

@2
sequential/conv3d_1/Relu?
$sequential/max_pooling3d_1/MaxPool3D	MaxPool3D&sequential/conv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:?????????@*
ksize	
*
paddingSAME*
strides	
2&
$sequential/max_pooling3d_1/MaxPool3D?
)sequential/conv3d_2/Conv3D/ReadVariableOpReadVariableOp2sequential_conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02+
)sequential/conv3d_2/Conv3D/ReadVariableOp?
sequential/conv3d_2/Conv3DConv3D-sequential/max_pooling3d_1/MaxPool3D:output:01sequential/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
2
sequential/conv3d_2/Conv3D?
*sequential/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv3d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*sequential/conv3d_2/BiasAdd/ReadVariableOp?
sequential/conv3d_2/BiasAddBiasAdd#sequential/conv3d_2/Conv3D:output:02sequential/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????2
sequential/conv3d_2/BiasAdd?
sequential/conv3d_2/ReluRelu$sequential/conv3d_2/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
sequential/conv3d_2/Relu?
$sequential/max_pooling3d_2/MaxPool3D	MaxPool3D&sequential/conv3d_2/Relu:activations:0*
T0*4
_output_shapes"
 :??????????*
ksize	
*
paddingSAME*
strides	
2&
$sequential/max_pooling3d_2/MaxPool3D?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? $  2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape-sequential/max_pooling3d_2/MaxPool3D:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????H2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?H
*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential/dense/BiasAdd?
IdentityIdentity!sequential/dense/BiasAdd:output:0)^sequential/conv3d/BiasAdd/ReadVariableOp(^sequential/conv3d/Conv3D/ReadVariableOp+^sequential/conv3d_1/BiasAdd/ReadVariableOp*^sequential/conv3d_1/Conv3D/ReadVariableOp+^sequential/conv3d_2/BiasAdd/ReadVariableOp*^sequential/conv3d_2/Conv3D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????<: : : : : : : : 2T
(sequential/conv3d/BiasAdd/ReadVariableOp(sequential/conv3d/BiasAdd/ReadVariableOp2R
'sequential/conv3d/Conv3D/ReadVariableOp'sequential/conv3d/Conv3D/ReadVariableOp2X
*sequential/conv3d_1/BiasAdd/ReadVariableOp*sequential/conv3d_1/BiasAdd/ReadVariableOp2V
)sequential/conv3d_1/Conv3D/ReadVariableOp)sequential/conv3d_1/Conv3D/ReadVariableOp2X
*sequential/conv3d_2/BiasAdd/ReadVariableOp*sequential/conv3d_2/BiasAdd/ReadVariableOp2V
)sequential/conv3d_2/Conv3D/ReadVariableOp)sequential/conv3d_2/Conv3D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:a ]
3
_output_shapes!
:?????????<
&
_user_specified_nameconv3d_input
?	
?
+__inference_sequential_layer_call_fn_423481
conv3d_input%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@?
	unknown_4:	?
	unknown_5:	?H

	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv3d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_4234622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????<: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
3
_output_shapes!
:?????????<
&
_user_specified_nameconv3d_input
?
?
)__inference_conv3d_1_layer_call_fn_423851

inputs%
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????

@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_4234122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????

@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????

 : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????

 
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_423877

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? $  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????H2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?$
?
F__inference_sequential_layer_call_and_return_conditional_losses_423578

inputs+
conv3d_423553: 
conv3d_423555: -
conv3d_1_423559: @
conv3d_1_423561:@.
conv3d_2_423565:@?
conv3d_2_423567:	?
dense_423572:	?H

dense_423574:

identity??conv3d/StatefulPartitionedCall? conv3d_1/StatefulPartitionedCall? conv3d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_423553conv3d_423555*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????< *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_4233942 
conv3d/StatefulPartitionedCall?
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????

 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_4233462
max_pooling3d/PartitionedCall?
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_1_423559conv3d_1_423561*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????

@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_4234122"
 conv3d_1/StatefulPartitionedCall?
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_4233582!
max_pooling3d_1/PartitionedCall?
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_2_423565conv3d_2_423567*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_4234302"
 conv3d_2/StatefulPartitionedCall?
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_4233702!
max_pooling3d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling3d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_4234432
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_423572dense_423574*
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
GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4234552
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????<: : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:[ W
3
_output_shapes!
:?????????<
 
_user_specified_nameinputs
?	
?
+__inference_sequential_layer_call_fn_423618
conv3d_input%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@?
	unknown_4:	?
	unknown_5:	?H

	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv3d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_4235782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????<: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
3
_output_shapes!
:?????????<
&
_user_specified_nameconv3d_input
?
?
__inference__traced_save_423948
file_prefix,
(savev2_conv3d_kernel_read_readvariableop*
&savev2_conv3d_bias_read_readvariableop.
*savev2_conv3d_1_kernel_read_readvariableop,
(savev2_conv3d_1_bias_read_readvariableop.
*savev2_conv3d_2_kernel_read_readvariableop,
(savev2_conv3d_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv3d_kernel_read_readvariableop&savev2_conv3d_bias_read_readvariableop*savev2_conv3d_1_kernel_read_readvariableop(savev2_conv3d_1_bias_read_readvariableop*savev2_conv3d_2_kernel_read_readvariableop(savev2_conv3d_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*~
_input_shapesm
k: : : : @:@:@?:?:	?H
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
: @: 

_output_shapes
:@:1-
+
_output_shapes
:@?:!

_output_shapes	
:?:%!

_output_shapes
:	?H
: 

_output_shapes
:
:	

_output_shapes
: 
?
?
'__inference_conv3d_layer_call_fn_423831

inputs%
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????< *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_4233942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????< 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????<: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????<
 
_user_specified_nameinputs
?$
?
F__inference_sequential_layer_call_and_return_conditional_losses_423646
conv3d_input+
conv3d_423621: 
conv3d_423623: -
conv3d_1_423627: @
conv3d_1_423629:@.
conv3d_2_423633:@?
conv3d_2_423635:	?
dense_423640:	?H

dense_423642:

identity??conv3d/StatefulPartitionedCall? conv3d_1/StatefulPartitionedCall? conv3d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv3d/StatefulPartitionedCallStatefulPartitionedCallconv3d_inputconv3d_423621conv3d_423623*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????< *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_4233942 
conv3d/StatefulPartitionedCall?
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????

 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_4233462
max_pooling3d/PartitionedCall?
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_1_423627conv3d_1_423629*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????

@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_4234122"
 conv3d_1/StatefulPartitionedCall?
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_4233582!
max_pooling3d_1/PartitionedCall?
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_2_423633conv3d_2_423635*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_4234302"
 conv3d_2/StatefulPartitionedCall?
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_4233702!
max_pooling3d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling3d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_4234432
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_423640dense_423642*
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
GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4234552
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????<: : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:a ]
3
_output_shapes!
:?????????<
&
_user_specified_nameconv3d_input
?	
?
A__inference_dense_layer_call_and_return_conditional_losses_423455

inputs1
matmul_readvariableop_resource:	?H
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?H
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
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????H
 
_user_specified_nameinputs
?
L
0__inference_max_pooling3d_2_layer_call_fn_423376

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_4233702
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_423370

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingSAME*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?$
?
F__inference_sequential_layer_call_and_return_conditional_losses_423674
conv3d_input+
conv3d_423649: 
conv3d_423651: -
conv3d_1_423655: @
conv3d_1_423657:@.
conv3d_2_423661:@?
conv3d_2_423663:	?
dense_423668:	?H

dense_423670:

identity??conv3d/StatefulPartitionedCall? conv3d_1/StatefulPartitionedCall? conv3d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv3d/StatefulPartitionedCallStatefulPartitionedCallconv3d_inputconv3d_423649conv3d_423651*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????< *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_4233942 
conv3d/StatefulPartitionedCall?
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????

 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_4233462
max_pooling3d/PartitionedCall?
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_1_423655conv3d_1_423657*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????

@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_4234122"
 conv3d_1/StatefulPartitionedCall?
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_4233582!
max_pooling3d_1/PartitionedCall?
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_2_423661conv3d_2_423663*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_4234302"
 conv3d_2/StatefulPartitionedCall?
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_4233702!
max_pooling3d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling3d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_4234432
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_423668dense_423670*
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
GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4234552
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????<: : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:a ]
3
_output_shapes!
:?????????<
&
_user_specified_nameconv3d_input
?
?
B__inference_conv3d_layer_call_and_return_conditional_losses_423822

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????< *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????< 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????< 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????< 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????<
 
_user_specified_nameinputs
?	
?
$__inference_signature_wrapper_423697
conv3d_input%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@?
	unknown_4:	?
	unknown_5:	?H

	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv3d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU2*0J 8? **
f%R#
!__inference__wrapped_model_4233402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????<: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
3
_output_shapes!
:?????????<
&
_user_specified_nameconv3d_input
?
e
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_423346

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingSAME*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_max_pooling3d_layer_call_fn_423352

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_4233462
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv3d_1_layer_call_and_return_conditional_losses_423842

inputs<
conv3d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

@*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

@2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????

@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????

@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????

 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????

 
 
_user_specified_nameinputs
?
?
D__inference_conv3d_1_layer_call_and_return_conditional_losses_423412

inputs<
conv3d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

@*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

@2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????

@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????

@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????

 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????

 
 
_user_specified_nameinputs
?
?
)__inference_conv3d_2_layer_call_fn_423871

inputs&
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_4234302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
?
B__inference_conv3d_layer_call_and_return_conditional_losses_423394

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????< *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????< 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????< 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????< 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????<
 
_user_specified_nameinputs
?
D
(__inference_flatten_layer_call_fn_423882

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
:??????????H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_4234432
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?
?
D__inference_conv3d_2_layer_call_and_return_conditional_losses_423862

inputs=
conv3d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*4
_output_shapes"
 :??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
?
D__inference_conv3d_2_layer_call_and_return_conditional_losses_423430

inputs=
conv3d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*4
_output_shapes"
 :??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?$
?
F__inference_sequential_layer_call_and_return_conditional_losses_423462

inputs+
conv3d_423395: 
conv3d_423397: -
conv3d_1_423413: @
conv3d_1_423415:@.
conv3d_2_423431:@?
conv3d_2_423433:	?
dense_423456:	?H

dense_423458:

identity??conv3d/StatefulPartitionedCall? conv3d_1/StatefulPartitionedCall? conv3d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_423395conv3d_423397*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????< *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_4233942 
conv3d/StatefulPartitionedCall?
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????

 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_4233462
max_pooling3d/PartitionedCall?
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_1_423413conv3d_1_423415*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????

@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_4234122"
 conv3d_1/StatefulPartitionedCall?
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_4233582!
max_pooling3d_1/PartitionedCall?
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_2_423431conv3d_2_423433*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_4234302"
 conv3d_2/StatefulPartitionedCall?
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_4233702!
max_pooling3d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling3d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_4234432
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_423456dense_423458*
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
GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4234552
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????<: : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:[ W
3
_output_shapes!
:?????????<
 
_user_specified_nameinputs
?	
?
+__inference_sequential_layer_call_fn_423790

inputs%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@?
	unknown_4:	?
	unknown_5:	?H
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
GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_4234622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????<: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????<
 
_user_specified_nameinputs
?	
?
+__inference_sequential_layer_call_fn_423811

inputs%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@?
	unknown_4:	?
	unknown_5:	?H
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
GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_4235782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????<: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????<
 
_user_specified_nameinputs
?&
?
"__inference__traced_restore_423982
file_prefix<
assignvariableop_conv3d_kernel: ,
assignvariableop_1_conv3d_bias: @
"assignvariableop_2_conv3d_1_kernel: @.
 assignvariableop_3_conv3d_1_bias:@A
"assignvariableop_4_conv3d_2_kernel:@?/
 assignvariableop_5_conv3d_2_bias:	?2
assignvariableop_6_dense_kernel:	?H
+
assignvariableop_7_dense_bias:
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
AssignVariableOpAssignVariableOpassignvariableop_conv3d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv3d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv3d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv3d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv3d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv3d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
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
?/
?
F__inference_sequential_layer_call_and_return_conditional_losses_423769

inputsC
%conv3d_conv3d_readvariableop_resource: 4
&conv3d_biasadd_readvariableop_resource: E
'conv3d_1_conv3d_readvariableop_resource: @6
(conv3d_1_biasadd_readvariableop_resource:@F
'conv3d_2_conv3d_readvariableop_resource:@?7
(conv3d_2_biasadd_readvariableop_resource:	?7
$dense_matmul_readvariableop_resource:	?H
3
%dense_biasadd_readvariableop_resource:

identity??conv3d/BiasAdd/ReadVariableOp?conv3d/Conv3D/ReadVariableOp?conv3d_1/BiasAdd/ReadVariableOp?conv3d_1/Conv3D/ReadVariableOp?conv3d_2/BiasAdd/ReadVariableOp?conv3d_2/Conv3D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02
conv3d/Conv3D/ReadVariableOp?
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????< *
paddingSAME*
strides	
2
conv3d/Conv3D?
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3d/BiasAdd/ReadVariableOp?
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????< 2
conv3d/BiasAddy
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????< 2
conv3d/Relu?
max_pooling3d/MaxPool3D	MaxPool3Dconv3d/Relu:activations:0*
T0*3
_output_shapes!
:?????????

 *
ksize	
*
paddingSAME*
strides	
2
max_pooling3d/MaxPool3D?
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02 
conv3d_1/Conv3D/ReadVariableOp?
conv3d_1/Conv3DConv3D max_pooling3d/MaxPool3D:output:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

@*
paddingSAME*
strides	
2
conv3d_1/Conv3D?
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_1/BiasAdd/ReadVariableOp?
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

@2
conv3d_1/BiasAdd
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????

@2
conv3d_1/Relu?
max_pooling3d_1/MaxPool3D	MaxPool3Dconv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:?????????@*
ksize	
*
paddingSAME*
strides	
2
max_pooling3d_1/MaxPool3D?
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02 
conv3d_2/Conv3D/ReadVariableOp?
conv3d_2/Conv3DConv3D"max_pooling3d_1/MaxPool3D:output:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
2
conv3d_2/Conv3D?
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv3d_2/BiasAdd/ReadVariableOp?
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????2
conv3d_2/BiasAdd?
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
conv3d_2/Relu?
max_pooling3d_2/MaxPool3D	MaxPool3Dconv3d_2/Relu:activations:0*
T0*4
_output_shapes"
 :??????????*
ksize	
*
paddingSAME*
strides	
2
max_pooling3d_2/MaxPool3Do
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? $  2
flatten/Const?
flatten/ReshapeReshape"max_pooling3d_2/MaxPool3D:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????H2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?H
*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense/BiasAdd?
IdentityIdentitydense/BiasAdd:output:0^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????<: : : : : : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:?????????<
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Q
conv3d_inputA
serving_default_conv3d_input:0?????????<9
dense0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?I
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	
signatures
#
_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
l_default_save_signature
*m&call_and_return_all_conditional_losses
n__call__"?F
_tf_keras_sequential?E{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 20, 20, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv3d_input"}}, {"class_name": "Conv3D", "config": {"name": "conv3d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 20, 20, 4]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv3D", "config": {"name": "conv3d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv3D", "config": {"name": "conv3d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 4}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 20, 20, 4]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 60, 20, 20, 4]}, "float32", "conv3d_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 20, 20, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv3d_input"}, "shared_object_id": 0}, {"class_name": "Conv3D", "config": {"name": "conv3d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 20, 20, 4]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 4}, {"class_name": "Conv3D", "config": {"name": "conv3d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7}, {"class_name": "Conv3D", "config": {"name": "conv3d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 10}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 11}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14}]}}}
?

kernel
bias
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
*o&call_and_return_all_conditional_losses
p__call__"?

_tf_keras_layer?
{"name": "conv3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 20, 20, 4]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 20, 20, 4]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 4}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 20, 20, 4]}}
?
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
*q&call_and_return_all_conditional_losses
r__call__"?
_tf_keras_layer?{"name": "max_pooling3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling3D", "config": {"name": "max_pooling3d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 17}}
?

kernel
bias
#_self_saveable_object_factories
trainable_variables
regularization_losses
 	variables
!	keras_api
*s&call_and_return_all_conditional_losses
t__call__"?	
_tf_keras_layer?	{"name": "conv3d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 32}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 10, 10, 32]}}
?
#"_self_saveable_object_factories
#trainable_variables
$regularization_losses
%	variables
&	keras_api
*u&call_and_return_all_conditional_losses
v__call__"?
_tf_keras_layer?{"name": "max_pooling3d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 19}}
?

'kernel
(bias
#)_self_saveable_object_factories
*trainable_variables
+regularization_losses
,	variables
-	keras_api
*w&call_and_return_all_conditional_losses
x__call__"?	
_tf_keras_layer?	{"name": "conv3d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 64}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 5, 5, 64]}}
?
#._self_saveable_object_factories
/trainable_variables
0regularization_losses
1	variables
2	keras_api
*y&call_and_return_all_conditional_losses
z__call__"?
_tf_keras_layer?{"name": "max_pooling3d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 21}}
?
#3_self_saveable_object_factories
4trainable_variables
5regularization_losses
6	variables
7	keras_api
*{&call_and_return_all_conditional_losses
|__call__"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 22}}
?

8kernel
9bias
#:_self_saveable_object_factories
;trainable_variables
<regularization_losses
=	variables
>	keras_api
*}&call_and_return_all_conditional_losses
~__call__"?
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9216}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9216]}}
,
serving_default"
signature_map
 "
trackable_dict_wrapper
X
0
1
2
3
'4
(5
86
97"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
'4
(5
86
97"
trackable_list_wrapper
?
trainable_variables
regularization_losses
?non_trainable_variables
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables

Clayers
n__call__
l_default_save_signature
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
+:) 2conv3d/kernel
: 2conv3d/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
trainable_variables
regularization_losses
Dnon_trainable_variables
Elayer_regularization_losses
Fmetrics
Glayer_metrics
	variables

Hlayers
p__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
regularization_losses
Inon_trainable_variables
Jlayer_regularization_losses
Kmetrics
Llayer_metrics
	variables

Mlayers
r__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
-:+ @2conv3d_1/kernel
:@2conv3d_1/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
trainable_variables
regularization_losses
Nnon_trainable_variables
Olayer_regularization_losses
Pmetrics
Qlayer_metrics
 	variables

Rlayers
t__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
#trainable_variables
$regularization_losses
Snon_trainable_variables
Tlayer_regularization_losses
Umetrics
Vlayer_metrics
%	variables

Wlayers
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
.:,@?2conv3d_2/kernel
:?2conv3d_2/bias
 "
trackable_dict_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
*trainable_variables
+regularization_losses
Xnon_trainable_variables
Ylayer_regularization_losses
Zmetrics
[layer_metrics
,	variables

\layers
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
/trainable_variables
0regularization_losses
]non_trainable_variables
^layer_regularization_losses
_metrics
`layer_metrics
1	variables

alayers
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
4trainable_variables
5regularization_losses
bnon_trainable_variables
clayer_regularization_losses
dmetrics
elayer_metrics
6	variables

flayers
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
:	?H
2dense/kernel
:
2
dense/bias
 "
trackable_dict_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
?
;trainable_variables
<regularization_losses
gnon_trainable_variables
hlayer_regularization_losses
imetrics
jlayer_metrics
=	variables

klayers
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
7"
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
?2?
!__inference__wrapped_model_423340?
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
annotations? *7?4
2?/
conv3d_input?????????<
?2?
F__inference_sequential_layer_call_and_return_conditional_losses_423733
F__inference_sequential_layer_call_and_return_conditional_losses_423769
F__inference_sequential_layer_call_and_return_conditional_losses_423646
F__inference_sequential_layer_call_and_return_conditional_losses_423674?
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
?2?
+__inference_sequential_layer_call_fn_423481
+__inference_sequential_layer_call_fn_423790
+__inference_sequential_layer_call_fn_423811
+__inference_sequential_layer_call_fn_423618?
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
B__inference_conv3d_layer_call_and_return_conditional_losses_423822?
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
'__inference_conv3d_layer_call_fn_423831?
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
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_423346?
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
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
.__inference_max_pooling3d_layer_call_fn_423352?
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
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
D__inference_conv3d_1_layer_call_and_return_conditional_losses_423842?
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
)__inference_conv3d_1_layer_call_fn_423851?
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
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_423358?
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
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
0__inference_max_pooling3d_1_layer_call_fn_423364?
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
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
D__inference_conv3d_2_layer_call_and_return_conditional_losses_423862?
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
)__inference_conv3d_2_layer_call_fn_423871?
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
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_423370?
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
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
0__inference_max_pooling3d_2_layer_call_fn_423376?
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
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
C__inference_flatten_layer_call_and_return_conditional_losses_423877?
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
(__inference_flatten_layer_call_fn_423882?
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
A__inference_dense_layer_call_and_return_conditional_losses_423892?
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
&__inference_dense_layer_call_fn_423901?
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
$__inference_signature_wrapper_423697conv3d_input"?
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
!__inference__wrapped_model_423340|'(89A?>
7?4
2?/
conv3d_input?????????<
? "-?*
(
dense?
dense?????????
?
D__inference_conv3d_1_layer_call_and_return_conditional_losses_423842t;?8
1?.
,?)
inputs?????????

 
? "1?.
'?$
0?????????

@
? ?
)__inference_conv3d_1_layer_call_fn_423851g;?8
1?.
,?)
inputs?????????

 
? "$?!?????????

@?
D__inference_conv3d_2_layer_call_and_return_conditional_losses_423862u'(;?8
1?.
,?)
inputs?????????@
? "2?/
(?%
0??????????
? ?
)__inference_conv3d_2_layer_call_fn_423871h'(;?8
1?.
,?)
inputs?????????@
? "%?"???????????
B__inference_conv3d_layer_call_and_return_conditional_losses_423822t;?8
1?.
,?)
inputs?????????<
? "1?.
'?$
0?????????< 
? ?
'__inference_conv3d_layer_call_fn_423831g;?8
1?.
,?)
inputs?????????<
? "$?!?????????< ?
A__inference_dense_layer_call_and_return_conditional_losses_423892]890?-
&?#
!?
inputs??????????H
? "%?"
?
0?????????

? z
&__inference_dense_layer_call_fn_423901P890?-
&?#
!?
inputs??????????H
? "??????????
?
C__inference_flatten_layer_call_and_return_conditional_losses_423877f<?9
2?/
-?*
inputs??????????
? "&?#
?
0??????????H
? ?
(__inference_flatten_layer_call_fn_423882Y<?9
2?/
-?*
inputs??????????
? "???????????H?
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_423358?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
0__inference_max_pooling3d_1_layer_call_fn_423364?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_423370?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
0__inference_max_pooling3d_2_layer_call_fn_423376?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_423346?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
.__inference_max_pooling3d_layer_call_fn_423352?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
F__inference_sequential_layer_call_and_return_conditional_losses_423646|'(89I?F
??<
2?/
conv3d_input?????????<
p 

 
? "%?"
?
0?????????

? ?
F__inference_sequential_layer_call_and_return_conditional_losses_423674|'(89I?F
??<
2?/
conv3d_input?????????<
p

 
? "%?"
?
0?????????

? ?
F__inference_sequential_layer_call_and_return_conditional_losses_423733v'(89C?@
9?6
,?)
inputs?????????<
p 

 
? "%?"
?
0?????????

? ?
F__inference_sequential_layer_call_and_return_conditional_losses_423769v'(89C?@
9?6
,?)
inputs?????????<
p

 
? "%?"
?
0?????????

? ?
+__inference_sequential_layer_call_fn_423481o'(89I?F
??<
2?/
conv3d_input?????????<
p 

 
? "??????????
?
+__inference_sequential_layer_call_fn_423618o'(89I?F
??<
2?/
conv3d_input?????????<
p

 
? "??????????
?
+__inference_sequential_layer_call_fn_423790i'(89C?@
9?6
,?)
inputs?????????<
p 

 
? "??????????
?
+__inference_sequential_layer_call_fn_423811i'(89C?@
9?6
,?)
inputs?????????<
p

 
? "??????????
?
$__inference_signature_wrapper_423697?'(89Q?N
? 
G?D
B
conv3d_input2?/
conv3d_input?????????<"-?*
(
dense?
dense?????????
