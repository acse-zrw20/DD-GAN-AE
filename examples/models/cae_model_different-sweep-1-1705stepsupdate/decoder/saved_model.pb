??
??
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
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
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
?H*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	
?H*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?H*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?H*
dtype0
?
conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:??* 
shared_nameconv3d_3/kernel
?
#conv3d_3/kernel/Read/ReadVariableOpReadVariableOpconv3d_3/kernel*,
_output_shapes
:??*
dtype0
s
conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3d_3/bias
l
!conv3d_3/bias/Read/ReadVariableOpReadVariableOpconv3d_3/bias*
_output_shapes	
:?*
dtype0
?
conv3d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:?@* 
shared_nameconv3d_4/kernel
?
#conv3d_4/kernel/Read/ReadVariableOpReadVariableOpconv3d_4/kernel*+
_output_shapes
:?@*
dtype0
r
conv3d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d_4/bias
k
!conv3d_4/bias/Read/ReadVariableOpReadVariableOpconv3d_4/bias*
_output_shapes
:@*
dtype0
?
conv3d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv3d_5/kernel

#conv3d_5/kernel/Read/ReadVariableOpReadVariableOpconv3d_5/kernel**
_output_shapes
:@ *
dtype0
r
conv3d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_5/bias
k
!conv3d_5/bias/Read/ReadVariableOpReadVariableOpconv3d_5/bias*
_output_shapes
: *
dtype0
?
conv3d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv3d_6/kernel

#conv3d_6/kernel/Read/ReadVariableOpReadVariableOpconv3d_6/kernel**
_output_shapes
: *
dtype0
r
conv3d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_6/bias
k
!conv3d_6/bias/Read/ReadVariableOpReadVariableOpconv3d_6/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?(
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?(
value?(B?( B?(
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9

signatures
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
?

kernel
bias
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
w
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
?

kernel
bias
#_self_saveable_object_factories
 	variables
!regularization_losses
"trainable_variables
#	keras_api
w
#$_self_saveable_object_factories
%	variables
&regularization_losses
'trainable_variables
(	keras_api
?

)kernel
*bias
#+_self_saveable_object_factories
,	variables
-regularization_losses
.trainable_variables
/	keras_api
w
#0_self_saveable_object_factories
1	variables
2regularization_losses
3trainable_variables
4	keras_api
?

5kernel
6bias
#7_self_saveable_object_factories
8	variables
9regularization_losses
:trainable_variables
;	keras_api
w
#<_self_saveable_object_factories
=	variables
>regularization_losses
?trainable_variables
@	keras_api
?

Akernel
Bbias
#C_self_saveable_object_factories
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
w
#H_self_saveable_object_factories
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
 
 
F
0
1
2
3
)4
*5
56
67
A8
B9
 
F
0
1
2
3
)4
*5
56
67
A8
B9
?
Mlayer_metrics
	variables

Nlayers
Onon_trainable_variables
Pmetrics
regularization_losses
trainable_variables
Qlayer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

0
1
?
Rlayer_metrics
	variables

Slayers
Tnon_trainable_variables
Umetrics
regularization_losses
trainable_variables
Vlayer_regularization_losses
 
 
 
 
?
Wlayer_metrics
	variables

Xlayers
Ynon_trainable_variables
Zmetrics
regularization_losses
trainable_variables
[layer_regularization_losses
[Y
VARIABLE_VALUEconv3d_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

0
1
?
\layer_metrics
 	variables

]layers
^non_trainable_variables
_metrics
!regularization_losses
"trainable_variables
`layer_regularization_losses
 
 
 
 
?
alayer_metrics
%	variables

blayers
cnon_trainable_variables
dmetrics
&regularization_losses
'trainable_variables
elayer_regularization_losses
[Y
VARIABLE_VALUEconv3d_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1
 

)0
*1
?
flayer_metrics
,	variables

glayers
hnon_trainable_variables
imetrics
-regularization_losses
.trainable_variables
jlayer_regularization_losses
 
 
 
 
?
klayer_metrics
1	variables

llayers
mnon_trainable_variables
nmetrics
2regularization_losses
3trainable_variables
olayer_regularization_losses
[Y
VARIABLE_VALUEconv3d_5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

50
61
 

50
61
?
player_metrics
8	variables

qlayers
rnon_trainable_variables
smetrics
9regularization_losses
:trainable_variables
tlayer_regularization_losses
 
 
 
 
?
ulayer_metrics
=	variables

vlayers
wnon_trainable_variables
xmetrics
>regularization_losses
?trainable_variables
ylayer_regularization_losses
[Y
VARIABLE_VALUEconv3d_6/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_6/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

A0
B1
 

A0
B1
?
zlayer_metrics
D	variables

{layers
|non_trainable_variables
}metrics
Eregularization_losses
Ftrainable_variables
~layer_regularization_losses
 
 
 
 
?
layer_metrics
I	variables
?layers
?non_trainable_variables
?metrics
Jregularization_losses
Ktrainable_variables
 ?layer_regularization_losses
 
F
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
?
serving_default_dense_1_inputPlaceholder*'
_output_shapes
:?????????
*
dtype0*
shape:?????????

?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_1_inputdense_1/kerneldense_1/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/biasconv3d_6/kernelconv3d_6/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????<*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_3798070
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp#conv3d_3/kernel/Read/ReadVariableOp!conv3d_3/bias/Read/ReadVariableOp#conv3d_4/kernel/Read/ReadVariableOp!conv3d_4/bias/Read/ReadVariableOp#conv3d_5/kernel/Read/ReadVariableOp!conv3d_5/bias/Read/ReadVariableOp#conv3d_6/kernel/Read/ReadVariableOp!conv3d_6/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU2*0J 8? *)
f$R"
 __inference__traced_save_3798785
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/biasconv3d_6/kernelconv3d_6/bias*
Tin
2*
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
GPU2*0J 8? *,
f'R%
#__inference__traced_restore_3798825˦

?
?
E__inference_conv3d_3_layer_call_and_return_conditional_losses_3797573

inputs>
conv3d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
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
 :??????????2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*4
_output_shapes"
 :??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?
E
)__inference_reshape_layer_call_fn_3798491

inputs
identity?
PartitionedCallPartitionedCallinputs*
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
GPU2*0J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_37975602
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????H:P L
(
_output_shapes
:??????????H
 
_user_specified_nameinputs
?-
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3798043
dense_1_input"
dense_1_3798012:	
?H
dense_1_3798014:	?H0
conv3d_3_3798018:??
conv3d_3_3798020:	?/
conv3d_4_3798024:?@
conv3d_4_3798026:@.
conv3d_5_3798030:@ 
conv3d_5_3798032: .
conv3d_6_3798036: 
conv3d_6_3798038:
identity?? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall? conv3d_6/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCalldense_1_inputdense_1_3798012dense_1_3798014*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_37975392!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_37975602
reshape/PartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv3d_3_3798018conv3d_3_3798020*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_3_layer_call_and_return_conditional_losses_37975732"
 conv3d_3/StatefulPartitionedCall?
up_sampling3d/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_37976062
up_sampling3d/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall&up_sampling3d/PartitionedCall:output:0conv3d_4_3798024conv3d_4_3798026*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_4_layer_call_and_return_conditional_losses_37976192"
 conv3d_4/StatefulPartitionedCall?
up_sampling3d_1/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:????????? @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_37976662!
up_sampling3d_1/PartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_1/PartitionedCall:output:0conv3d_5_3798030conv3d_5_3798032*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_5_layer_call_and_return_conditional_losses_37976792"
 conv3d_5/StatefulPartitionedCall?
up_sampling3d_2/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????< * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_37977482!
up_sampling3d_2/PartitionedCall?
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_2/PartitionedCall:output:0conv3d_6_3798036conv3d_6_3798038*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_6_layer_call_and_return_conditional_losses_37977612"
 conv3d_6/StatefulPartitionedCall?
cropping3d/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_cropping3d_layer_call_and_return_conditional_losses_37975152
cropping3d/PartitionedCall?
IdentityIdentity#cropping3d/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
: : : : : : : : : : 2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:V R
'
_output_shapes
:?????????

'
_user_specified_namedense_1_input
?
?
E__inference_conv3d_6_layer_call_and_return_conditional_losses_3798732

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????<*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????<2	
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????<2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????< : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????< 
 
_user_specified_nameinputs
?
`
D__inference_reshape_layer_call_and_return_conditional_losses_3797560

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
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3e
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/4?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape|
ReshapeReshapeinputsReshape/shape:output:0*
T0*4
_output_shapes"
 :??????????2	
Reshapeq
IdentityIdentityReshape:output:0*
T0*4
_output_shapes"
 :??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????H:P L
(
_output_shapes
:??????????H
 
_user_specified_nameinputs
?

?
.__inference_sequential_1_layer_call_fn_3798095

inputs
unknown:	
?H
	unknown_0:	?H)
	unknown_1:??
	unknown_2:	?(
	unknown_3:?@
	unknown_4:@'
	unknown_5:@ 
	unknown_6: '
	unknown_7: 
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????<*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_37977692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?'
h
L__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_3797748

inputs
identityd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 *
	num_split2
split\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15split:output:16split:output:16split:output:17split:output:17split:output:18split:output:18split:output:19split:output:19split:output:20split:output:20split:output:21split:output:21split:output:22split:output:22split:output:23split:output:23split:output:24split:output:24split:output:25split:output:25split:output:26split:output:26split:output:27split:output:27split:output:28split:output:28split:output:29split:output:29concat/axis:output:0*
N<*
T0*3
_output_shapes!
:?????????<

 2
concath
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*?
_output_shapes?
?:?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 *
	num_split
2	
split_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:?????????<
 2

concat_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*?
_output_shapes?
?:?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< *
	num_split
2	
split_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis?
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:?????????< 2

concat_2q
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:?????????< 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????

 :[ W
3
_output_shapes!
:?????????

 
 
_user_specified_nameinputs
ι
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3798466

inputs9
&dense_1_matmul_readvariableop_resource:	
?H6
'dense_1_biasadd_readvariableop_resource:	?HG
'conv3d_3_conv3d_readvariableop_resource:??7
(conv3d_3_biasadd_readvariableop_resource:	?F
'conv3d_4_conv3d_readvariableop_resource:?@6
(conv3d_4_biasadd_readvariableop_resource:@E
'conv3d_5_conv3d_readvariableop_resource:@ 6
(conv3d_5_biasadd_readvariableop_resource: E
'conv3d_6_conv3d_readvariableop_resource: 6
(conv3d_6_biasadd_readvariableop_resource:
identity??conv3d_3/BiasAdd/ReadVariableOp?conv3d_3/Conv3D/ReadVariableOp?conv3d_4/BiasAdd/ReadVariableOp?conv3d_4/Conv3D/ReadVariableOp?conv3d_5/BiasAdd/ReadVariableOp?conv3d_5/Conv3D/ReadVariableOp?conv3d_6/BiasAdd/ReadVariableOp?conv3d_6/Conv3D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	
?H*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????H2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?H*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????H2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????H2
dense_1/Reluh
reshape/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3u
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/4?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_1/Relu:activations:0reshape/Reshape/shape:output:0*
T0*4
_output_shapes"
 :??????????2
reshape/Reshape?
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02 
conv3d_3/Conv3D/ReadVariableOp?
conv3d_3/Conv3DConv3Dreshape/Reshape:output:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
2
conv3d_3/Conv3D?
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv3d_3/BiasAdd/ReadVariableOp?
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????2
conv3d_3/BiasAdd?
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
conv3d_3/Relu?
up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/split/split_dim?
up_sampling3d/splitSplit&up_sampling3d/split/split_dim:output:0conv3d_3/Relu:activations:0*
T0*?
_output_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*
	num_split2
up_sampling3d/splitx
up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/concat/axis?
up_sampling3d/concatConcatV2up_sampling3d/split:output:0up_sampling3d/split:output:0up_sampling3d/split:output:1up_sampling3d/split:output:1up_sampling3d/split:output:2up_sampling3d/split:output:2up_sampling3d/split:output:3up_sampling3d/split:output:3up_sampling3d/split:output:4up_sampling3d/split:output:4up_sampling3d/split:output:5up_sampling3d/split:output:5up_sampling3d/split:output:6up_sampling3d/split:output:6up_sampling3d/split:output:7up_sampling3d/split:output:7"up_sampling3d/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2
up_sampling3d/concat?
up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d/split_1/split_dim?
up_sampling3d/split_1Split(up_sampling3d/split_1/split_dim:output:0up_sampling3d/concat:output:0*
T0*t
_output_shapesb
`:??????????:??????????:??????????*
	num_split2
up_sampling3d/split_1|
up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/concat_1/axis?
up_sampling3d/concat_1ConcatV2up_sampling3d/split_1:output:0up_sampling3d/split_1:output:0up_sampling3d/split_1:output:1up_sampling3d/split_1:output:1up_sampling3d/split_1:output:2up_sampling3d/split_1:output:2$up_sampling3d/concat_1/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2
up_sampling3d/concat_1?
up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d/split_2/split_dim?
up_sampling3d/split_2Split(up_sampling3d/split_2/split_dim:output:0up_sampling3d/concat_1:output:0*
T0*t
_output_shapesb
`:??????????:??????????:??????????*
	num_split2
up_sampling3d/split_2|
up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/concat_2/axis?
up_sampling3d/concat_2ConcatV2up_sampling3d/split_2:output:0up_sampling3d/split_2:output:0up_sampling3d/split_2:output:1up_sampling3d/split_2:output:1up_sampling3d/split_2:output:2up_sampling3d/split_2:output:2$up_sampling3d/concat_2/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2
up_sampling3d/concat_2?
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource*+
_output_shapes
:?@*
dtype02 
conv3d_4/Conv3D/ReadVariableOp?
conv3d_4/Conv3DConv3Dup_sampling3d/concat_2:output:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
2
conv3d_4/Conv3D?
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_4/BiasAdd/ReadVariableOp?
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@2
conv3d_4/BiasAdd
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@2
conv3d_4/Relu?
up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d_1/split/split_dim?
up_sampling3d_1/splitSplit(up_sampling3d_1/split/split_dim:output:0conv3d_4/Relu:activations:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
up_sampling3d_1/split|
up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/concat/axis?	
up_sampling3d_1/concatConcatV2up_sampling3d_1/split:output:0up_sampling3d_1/split:output:0up_sampling3d_1/split:output:1up_sampling3d_1/split:output:1up_sampling3d_1/split:output:2up_sampling3d_1/split:output:2up_sampling3d_1/split:output:3up_sampling3d_1/split:output:3up_sampling3d_1/split:output:4up_sampling3d_1/split:output:4up_sampling3d_1/split:output:5up_sampling3d_1/split:output:5up_sampling3d_1/split:output:6up_sampling3d_1/split:output:6up_sampling3d_1/split:output:7up_sampling3d_1/split:output:7up_sampling3d_1/split:output:8up_sampling3d_1/split:output:8up_sampling3d_1/split:output:9up_sampling3d_1/split:output:9up_sampling3d_1/split:output:10up_sampling3d_1/split:output:10up_sampling3d_1/split:output:11up_sampling3d_1/split:output:11up_sampling3d_1/split:output:12up_sampling3d_1/split:output:12up_sampling3d_1/split:output:13up_sampling3d_1/split:output:13up_sampling3d_1/split:output:14up_sampling3d_1/split:output:14up_sampling3d_1/split:output:15up_sampling3d_1/split:output:15$up_sampling3d_1/concat/axis:output:0*
N *
T0*3
_output_shapes!
:????????? @2
up_sampling3d_1/concat?
!up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_1/split_1/split_dim?
up_sampling3d_1/split_1Split*up_sampling3d_1/split_1/split_dim:output:0up_sampling3d_1/concat:output:0*
T0*?
_output_shapes?
?:????????? @:????????? @:????????? @:????????? @:????????? @:????????? @*
	num_split2
up_sampling3d_1/split_1?
up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/concat_1/axis?
up_sampling3d_1/concat_1ConcatV2 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:5&up_sampling3d_1/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:????????? @2
up_sampling3d_1/concat_1?
!up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_1/split_2/split_dim?
up_sampling3d_1/split_2Split*up_sampling3d_1/split_2/split_dim:output:0!up_sampling3d_1/concat_1:output:0*
T0*?
_output_shapes?
?:????????? @:????????? @:????????? @:????????? @:????????? @:????????? @*
	num_split2
up_sampling3d_1/split_2?
up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/concat_2/axis?
up_sampling3d_1/concat_2ConcatV2 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:5&up_sampling3d_1/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:????????? @2
up_sampling3d_1/concat_2?
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:@ *
dtype02 
conv3d_5/Conv3D/ReadVariableOp?
conv3d_5/Conv3DConv3D!up_sampling3d_1/concat_2:output:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

 *
paddingVALID*
strides	
2
conv3d_5/Conv3D?
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv3d_5/BiasAdd/ReadVariableOp?
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

 2
conv3d_5/BiasAdd
conv3d_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????

 2
conv3d_5/Relu?
up_sampling3d_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d_2/split/split_dim?
up_sampling3d_2/splitSplit(up_sampling3d_2/split/split_dim:output:0conv3d_5/Relu:activations:0*
T0*?
_output_shapes?
?:?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 *
	num_split2
up_sampling3d_2/split|
up_sampling3d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_2/concat/axis?
up_sampling3d_2/concatConcatV2up_sampling3d_2/split:output:0up_sampling3d_2/split:output:0up_sampling3d_2/split:output:1up_sampling3d_2/split:output:1up_sampling3d_2/split:output:2up_sampling3d_2/split:output:2up_sampling3d_2/split:output:3up_sampling3d_2/split:output:3up_sampling3d_2/split:output:4up_sampling3d_2/split:output:4up_sampling3d_2/split:output:5up_sampling3d_2/split:output:5up_sampling3d_2/split:output:6up_sampling3d_2/split:output:6up_sampling3d_2/split:output:7up_sampling3d_2/split:output:7up_sampling3d_2/split:output:8up_sampling3d_2/split:output:8up_sampling3d_2/split:output:9up_sampling3d_2/split:output:9up_sampling3d_2/split:output:10up_sampling3d_2/split:output:10up_sampling3d_2/split:output:11up_sampling3d_2/split:output:11up_sampling3d_2/split:output:12up_sampling3d_2/split:output:12up_sampling3d_2/split:output:13up_sampling3d_2/split:output:13up_sampling3d_2/split:output:14up_sampling3d_2/split:output:14up_sampling3d_2/split:output:15up_sampling3d_2/split:output:15up_sampling3d_2/split:output:16up_sampling3d_2/split:output:16up_sampling3d_2/split:output:17up_sampling3d_2/split:output:17up_sampling3d_2/split:output:18up_sampling3d_2/split:output:18up_sampling3d_2/split:output:19up_sampling3d_2/split:output:19up_sampling3d_2/split:output:20up_sampling3d_2/split:output:20up_sampling3d_2/split:output:21up_sampling3d_2/split:output:21up_sampling3d_2/split:output:22up_sampling3d_2/split:output:22up_sampling3d_2/split:output:23up_sampling3d_2/split:output:23up_sampling3d_2/split:output:24up_sampling3d_2/split:output:24up_sampling3d_2/split:output:25up_sampling3d_2/split:output:25up_sampling3d_2/split:output:26up_sampling3d_2/split:output:26up_sampling3d_2/split:output:27up_sampling3d_2/split:output:27up_sampling3d_2/split:output:28up_sampling3d_2/split:output:28up_sampling3d_2/split:output:29up_sampling3d_2/split:output:29$up_sampling3d_2/concat/axis:output:0*
N<*
T0*3
_output_shapes!
:?????????<

 2
up_sampling3d_2/concat?
!up_sampling3d_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_2/split_1/split_dim?
up_sampling3d_2/split_1Split*up_sampling3d_2/split_1/split_dim:output:0up_sampling3d_2/concat:output:0*
T0*?
_output_shapes?
?:?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 *
	num_split
2
up_sampling3d_2/split_1?
up_sampling3d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_2/concat_1/axis?
up_sampling3d_2/concat_1ConcatV2 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:7 up_sampling3d_2/split_1:output:7 up_sampling3d_2/split_1:output:8 up_sampling3d_2/split_1:output:8 up_sampling3d_2/split_1:output:9 up_sampling3d_2/split_1:output:9&up_sampling3d_2/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:?????????<
 2
up_sampling3d_2/concat_1?
!up_sampling3d_2/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_2/split_2/split_dim?
up_sampling3d_2/split_2Split*up_sampling3d_2/split_2/split_dim:output:0!up_sampling3d_2/concat_1:output:0*
T0*?
_output_shapes?
?:?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< *
	num_split
2
up_sampling3d_2/split_2?
up_sampling3d_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_2/concat_2/axis?
up_sampling3d_2/concat_2ConcatV2 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:7 up_sampling3d_2/split_2:output:7 up_sampling3d_2/split_2:output:8 up_sampling3d_2/split_2:output:8 up_sampling3d_2/split_2:output:9 up_sampling3d_2/split_2:output:9&up_sampling3d_2/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:?????????< 2
up_sampling3d_2/concat_2?
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02 
conv3d_6/Conv3D/ReadVariableOp?
conv3d_6/Conv3DConv3D!up_sampling3d_2/concat_2:output:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????<*
paddingSAME*
strides	
2
conv3d_6/Conv3D?
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_6/BiasAdd/ReadVariableOp?
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????<2
conv3d_6/BiasAdd?
conv3d_6/SigmoidSigmoidconv3d_6/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????<2
conv3d_6/Sigmoid?
cropping3d/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    2 
cropping3d/strided_slice/stack?
 cropping3d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    2"
 cropping3d/strided_slice/stack_1?
 cropping3d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               2"
 cropping3d/strided_slice/stack_2?
cropping3d/strided_sliceStridedSliceconv3d_6/Sigmoid:y:0'cropping3d/strided_slice/stack:output:0)cropping3d/strided_slice/stack_1:output:0)cropping3d/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:?????????<*

begin_mask*
end_mask2
cropping3d/strided_slice?
IdentityIdentity!cropping3d/strided_slice:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*3
_output_shapes!
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
: : : : : : : : : : 2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp2B
conv3d_6/BiasAdd/ReadVariableOpconv3d_6/BiasAdd/ReadVariableOp2@
conv3d_6/Conv3D/ReadVariableOpconv3d_6/Conv3D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
ι
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3798293

inputs9
&dense_1_matmul_readvariableop_resource:	
?H6
'dense_1_biasadd_readvariableop_resource:	?HG
'conv3d_3_conv3d_readvariableop_resource:??7
(conv3d_3_biasadd_readvariableop_resource:	?F
'conv3d_4_conv3d_readvariableop_resource:?@6
(conv3d_4_biasadd_readvariableop_resource:@E
'conv3d_5_conv3d_readvariableop_resource:@ 6
(conv3d_5_biasadd_readvariableop_resource: E
'conv3d_6_conv3d_readvariableop_resource: 6
(conv3d_6_biasadd_readvariableop_resource:
identity??conv3d_3/BiasAdd/ReadVariableOp?conv3d_3/Conv3D/ReadVariableOp?conv3d_4/BiasAdd/ReadVariableOp?conv3d_4/Conv3D/ReadVariableOp?conv3d_5/BiasAdd/ReadVariableOp?conv3d_5/Conv3D/ReadVariableOp?conv3d_6/BiasAdd/ReadVariableOp?conv3d_6/Conv3D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	
?H*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????H2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?H*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????H2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????H2
dense_1/Reluh
reshape/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3u
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/4?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_1/Relu:activations:0reshape/Reshape/shape:output:0*
T0*4
_output_shapes"
 :??????????2
reshape/Reshape?
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02 
conv3d_3/Conv3D/ReadVariableOp?
conv3d_3/Conv3DConv3Dreshape/Reshape:output:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
2
conv3d_3/Conv3D?
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv3d_3/BiasAdd/ReadVariableOp?
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????2
conv3d_3/BiasAdd?
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
conv3d_3/Relu?
up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/split/split_dim?
up_sampling3d/splitSplit&up_sampling3d/split/split_dim:output:0conv3d_3/Relu:activations:0*
T0*?
_output_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*
	num_split2
up_sampling3d/splitx
up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/concat/axis?
up_sampling3d/concatConcatV2up_sampling3d/split:output:0up_sampling3d/split:output:0up_sampling3d/split:output:1up_sampling3d/split:output:1up_sampling3d/split:output:2up_sampling3d/split:output:2up_sampling3d/split:output:3up_sampling3d/split:output:3up_sampling3d/split:output:4up_sampling3d/split:output:4up_sampling3d/split:output:5up_sampling3d/split:output:5up_sampling3d/split:output:6up_sampling3d/split:output:6up_sampling3d/split:output:7up_sampling3d/split:output:7"up_sampling3d/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2
up_sampling3d/concat?
up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d/split_1/split_dim?
up_sampling3d/split_1Split(up_sampling3d/split_1/split_dim:output:0up_sampling3d/concat:output:0*
T0*t
_output_shapesb
`:??????????:??????????:??????????*
	num_split2
up_sampling3d/split_1|
up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/concat_1/axis?
up_sampling3d/concat_1ConcatV2up_sampling3d/split_1:output:0up_sampling3d/split_1:output:0up_sampling3d/split_1:output:1up_sampling3d/split_1:output:1up_sampling3d/split_1:output:2up_sampling3d/split_1:output:2$up_sampling3d/concat_1/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2
up_sampling3d/concat_1?
up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d/split_2/split_dim?
up_sampling3d/split_2Split(up_sampling3d/split_2/split_dim:output:0up_sampling3d/concat_1:output:0*
T0*t
_output_shapesb
`:??????????:??????????:??????????*
	num_split2
up_sampling3d/split_2|
up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/concat_2/axis?
up_sampling3d/concat_2ConcatV2up_sampling3d/split_2:output:0up_sampling3d/split_2:output:0up_sampling3d/split_2:output:1up_sampling3d/split_2:output:1up_sampling3d/split_2:output:2up_sampling3d/split_2:output:2$up_sampling3d/concat_2/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2
up_sampling3d/concat_2?
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource*+
_output_shapes
:?@*
dtype02 
conv3d_4/Conv3D/ReadVariableOp?
conv3d_4/Conv3DConv3Dup_sampling3d/concat_2:output:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
2
conv3d_4/Conv3D?
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_4/BiasAdd/ReadVariableOp?
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@2
conv3d_4/BiasAdd
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@2
conv3d_4/Relu?
up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d_1/split/split_dim?
up_sampling3d_1/splitSplit(up_sampling3d_1/split/split_dim:output:0conv3d_4/Relu:activations:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
up_sampling3d_1/split|
up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/concat/axis?	
up_sampling3d_1/concatConcatV2up_sampling3d_1/split:output:0up_sampling3d_1/split:output:0up_sampling3d_1/split:output:1up_sampling3d_1/split:output:1up_sampling3d_1/split:output:2up_sampling3d_1/split:output:2up_sampling3d_1/split:output:3up_sampling3d_1/split:output:3up_sampling3d_1/split:output:4up_sampling3d_1/split:output:4up_sampling3d_1/split:output:5up_sampling3d_1/split:output:5up_sampling3d_1/split:output:6up_sampling3d_1/split:output:6up_sampling3d_1/split:output:7up_sampling3d_1/split:output:7up_sampling3d_1/split:output:8up_sampling3d_1/split:output:8up_sampling3d_1/split:output:9up_sampling3d_1/split:output:9up_sampling3d_1/split:output:10up_sampling3d_1/split:output:10up_sampling3d_1/split:output:11up_sampling3d_1/split:output:11up_sampling3d_1/split:output:12up_sampling3d_1/split:output:12up_sampling3d_1/split:output:13up_sampling3d_1/split:output:13up_sampling3d_1/split:output:14up_sampling3d_1/split:output:14up_sampling3d_1/split:output:15up_sampling3d_1/split:output:15$up_sampling3d_1/concat/axis:output:0*
N *
T0*3
_output_shapes!
:????????? @2
up_sampling3d_1/concat?
!up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_1/split_1/split_dim?
up_sampling3d_1/split_1Split*up_sampling3d_1/split_1/split_dim:output:0up_sampling3d_1/concat:output:0*
T0*?
_output_shapes?
?:????????? @:????????? @:????????? @:????????? @:????????? @:????????? @*
	num_split2
up_sampling3d_1/split_1?
up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/concat_1/axis?
up_sampling3d_1/concat_1ConcatV2 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:5&up_sampling3d_1/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:????????? @2
up_sampling3d_1/concat_1?
!up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_1/split_2/split_dim?
up_sampling3d_1/split_2Split*up_sampling3d_1/split_2/split_dim:output:0!up_sampling3d_1/concat_1:output:0*
T0*?
_output_shapes?
?:????????? @:????????? @:????????? @:????????? @:????????? @:????????? @*
	num_split2
up_sampling3d_1/split_2?
up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/concat_2/axis?
up_sampling3d_1/concat_2ConcatV2 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:5&up_sampling3d_1/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:????????? @2
up_sampling3d_1/concat_2?
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:@ *
dtype02 
conv3d_5/Conv3D/ReadVariableOp?
conv3d_5/Conv3DConv3D!up_sampling3d_1/concat_2:output:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

 *
paddingVALID*
strides	
2
conv3d_5/Conv3D?
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv3d_5/BiasAdd/ReadVariableOp?
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

 2
conv3d_5/BiasAdd
conv3d_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????

 2
conv3d_5/Relu?
up_sampling3d_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d_2/split/split_dim?
up_sampling3d_2/splitSplit(up_sampling3d_2/split/split_dim:output:0conv3d_5/Relu:activations:0*
T0*?
_output_shapes?
?:?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 *
	num_split2
up_sampling3d_2/split|
up_sampling3d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_2/concat/axis?
up_sampling3d_2/concatConcatV2up_sampling3d_2/split:output:0up_sampling3d_2/split:output:0up_sampling3d_2/split:output:1up_sampling3d_2/split:output:1up_sampling3d_2/split:output:2up_sampling3d_2/split:output:2up_sampling3d_2/split:output:3up_sampling3d_2/split:output:3up_sampling3d_2/split:output:4up_sampling3d_2/split:output:4up_sampling3d_2/split:output:5up_sampling3d_2/split:output:5up_sampling3d_2/split:output:6up_sampling3d_2/split:output:6up_sampling3d_2/split:output:7up_sampling3d_2/split:output:7up_sampling3d_2/split:output:8up_sampling3d_2/split:output:8up_sampling3d_2/split:output:9up_sampling3d_2/split:output:9up_sampling3d_2/split:output:10up_sampling3d_2/split:output:10up_sampling3d_2/split:output:11up_sampling3d_2/split:output:11up_sampling3d_2/split:output:12up_sampling3d_2/split:output:12up_sampling3d_2/split:output:13up_sampling3d_2/split:output:13up_sampling3d_2/split:output:14up_sampling3d_2/split:output:14up_sampling3d_2/split:output:15up_sampling3d_2/split:output:15up_sampling3d_2/split:output:16up_sampling3d_2/split:output:16up_sampling3d_2/split:output:17up_sampling3d_2/split:output:17up_sampling3d_2/split:output:18up_sampling3d_2/split:output:18up_sampling3d_2/split:output:19up_sampling3d_2/split:output:19up_sampling3d_2/split:output:20up_sampling3d_2/split:output:20up_sampling3d_2/split:output:21up_sampling3d_2/split:output:21up_sampling3d_2/split:output:22up_sampling3d_2/split:output:22up_sampling3d_2/split:output:23up_sampling3d_2/split:output:23up_sampling3d_2/split:output:24up_sampling3d_2/split:output:24up_sampling3d_2/split:output:25up_sampling3d_2/split:output:25up_sampling3d_2/split:output:26up_sampling3d_2/split:output:26up_sampling3d_2/split:output:27up_sampling3d_2/split:output:27up_sampling3d_2/split:output:28up_sampling3d_2/split:output:28up_sampling3d_2/split:output:29up_sampling3d_2/split:output:29$up_sampling3d_2/concat/axis:output:0*
N<*
T0*3
_output_shapes!
:?????????<

 2
up_sampling3d_2/concat?
!up_sampling3d_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_2/split_1/split_dim?
up_sampling3d_2/split_1Split*up_sampling3d_2/split_1/split_dim:output:0up_sampling3d_2/concat:output:0*
T0*?
_output_shapes?
?:?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 *
	num_split
2
up_sampling3d_2/split_1?
up_sampling3d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_2/concat_1/axis?
up_sampling3d_2/concat_1ConcatV2 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:7 up_sampling3d_2/split_1:output:7 up_sampling3d_2/split_1:output:8 up_sampling3d_2/split_1:output:8 up_sampling3d_2/split_1:output:9 up_sampling3d_2/split_1:output:9&up_sampling3d_2/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:?????????<
 2
up_sampling3d_2/concat_1?
!up_sampling3d_2/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_2/split_2/split_dim?
up_sampling3d_2/split_2Split*up_sampling3d_2/split_2/split_dim:output:0!up_sampling3d_2/concat_1:output:0*
T0*?
_output_shapes?
?:?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< *
	num_split
2
up_sampling3d_2/split_2?
up_sampling3d_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_2/concat_2/axis?
up_sampling3d_2/concat_2ConcatV2 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:7 up_sampling3d_2/split_2:output:7 up_sampling3d_2/split_2:output:8 up_sampling3d_2/split_2:output:8 up_sampling3d_2/split_2:output:9 up_sampling3d_2/split_2:output:9&up_sampling3d_2/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:?????????< 2
up_sampling3d_2/concat_2?
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02 
conv3d_6/Conv3D/ReadVariableOp?
conv3d_6/Conv3DConv3D!up_sampling3d_2/concat_2:output:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????<*
paddingSAME*
strides	
2
conv3d_6/Conv3D?
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_6/BiasAdd/ReadVariableOp?
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????<2
conv3d_6/BiasAdd?
conv3d_6/SigmoidSigmoidconv3d_6/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????<2
conv3d_6/Sigmoid?
cropping3d/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    2 
cropping3d/strided_slice/stack?
 cropping3d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    2"
 cropping3d/strided_slice/stack_1?
 cropping3d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               2"
 cropping3d/strided_slice/stack_2?
cropping3d/strided_sliceStridedSliceconv3d_6/Sigmoid:y:0'cropping3d/strided_slice/stack:output:0)cropping3d/strided_slice/stack_1:output:0)cropping3d/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:?????????<*

begin_mask*
end_mask2
cropping3d/strided_slice?
IdentityIdentity!cropping3d/strided_slice:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*3
_output_shapes!
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
: : : : : : : : : : 2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp2B
conv3d_6/BiasAdd/ReadVariableOpconv3d_6/BiasAdd/ReadVariableOp2@
conv3d_6/Conv3D/ReadVariableOpconv3d_6/Conv3D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?'
h
L__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_3798712

inputs
identityd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 *
	num_split2
split\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15split:output:16split:output:16split:output:17split:output:17split:output:18split:output:18split:output:19split:output:19split:output:20split:output:20split:output:21split:output:21split:output:22split:output:22split:output:23split:output:23split:output:24split:output:24split:output:25split:output:25split:output:26split:output:26split:output:27split:output:27split:output:28split:output:28split:output:29split:output:29concat/axis:output:0*
N<*
T0*3
_output_shapes!
:?????????<

 2
concath
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*?
_output_shapes?
?:?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 *
	num_split
2	
split_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:?????????<
 2

concat_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*?
_output_shapes?
?:?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< *
	num_split
2	
split_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis?
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:?????????< 2

concat_2q
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:?????????< 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????

 :[ W
3
_output_shapes!
:?????????

 
 
_user_specified_nameinputs
?
h
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_3798624

inputs
identityd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
split\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15concat/axis:output:0*
N *
T0*3
_output_shapes!
:????????? @2
concath
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*?
_output_shapes?
?:????????? @:????????? @:????????? @:????????? @:????????? @:????????? @*
	num_split2	
split_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:????????? @2

concat_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*?
_output_shapes?
?:????????? @:????????? @:????????? @:????????? @:????????? @:????????? @*
	num_split2	
split_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis?
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:????????? @2

concat_2q
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:????????? @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?"
?
 __inference__traced_save_3798785
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop.
*savev2_conv3d_3_kernel_read_readvariableop,
(savev2_conv3d_3_bias_read_readvariableop.
*savev2_conv3d_4_kernel_read_readvariableop,
(savev2_conv3d_4_bias_read_readvariableop.
*savev2_conv3d_5_kernel_read_readvariableop,
(savev2_conv3d_5_bias_read_readvariableop.
*savev2_conv3d_6_kernel_read_readvariableop,
(savev2_conv3d_6_bias_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop*savev2_conv3d_3_kernel_read_readvariableop(savev2_conv3d_3_bias_read_readvariableop*savev2_conv3d_4_kernel_read_readvariableop(savev2_conv3d_4_bias_read_readvariableop*savev2_conv3d_5_kernel_read_readvariableop(savev2_conv3d_5_bias_read_readvariableop*savev2_conv3d_6_kernel_read_readvariableop(savev2_conv3d_6_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :	
?H:?H:??:?:?@:@:@ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	
?H:!

_output_shapes	
:?H:2.
,
_output_shapes
:??:!

_output_shapes	
:?:1-
+
_output_shapes
:?@: 

_output_shapes
:@:0,
*
_output_shapes
:@ : 

_output_shapes
: :0	,
*
_output_shapes
: : 


_output_shapes
::

_output_shapes
: 
?
h
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_3797666

inputs
identityd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
split\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15concat/axis:output:0*
N *
T0*3
_output_shapes!
:????????? @2
concath
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*?
_output_shapes?
?:????????? @:????????? @:????????? @:????????? @:????????? @:????????? @*
	num_split2	
split_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:????????? @2

concat_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*?
_output_shapes?
?:????????? @:????????? @:????????? @:????????? @:????????? @:????????? @*
	num_split2	
split_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis?
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:????????? @2

concat_2q
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:????????? @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_conv3d_4_layer_call_and_return_conditional_losses_3797619

inputs=
conv3d_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:?@*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
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
:?????????@2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?
M
1__inference_up_sampling3d_2_layer_call_fn_3798649

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????< * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_37977482
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????< 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????

 :[ W
3
_output_shapes!
:?????????

 
 
_user_specified_nameinputs
?
f
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_3798558

inputs
identityd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*
	num_split2
split\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2
concath
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*t
_output_shapesb
`:??????????:??????????:??????????*
	num_split2	
split_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2concat_1/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2

concat_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*t
_output_shapesb
`:??????????:??????????:??????????*
	num_split2	
split_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis?
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2concat_2/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2

concat_2r
IdentityIdentityconcat_2:output:0*
T0*4
_output_shapes"
 :??????????2

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
?
*__inference_conv3d_4_layer_call_fn_3798567

inputs&
unknown:?@
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
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_4_layer_call_and_return_conditional_losses_37976192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?
?
*__inference_conv3d_3_layer_call_fn_3798515

inputs'
unknown:??
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
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_3_layer_call_and_return_conditional_losses_37975732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?

?
.__inference_sequential_1_layer_call_fn_3797975
dense_1_input
unknown:	
?H
	unknown_0:	?H)
	unknown_1:??
	unknown_2:	?(
	unknown_3:?@
	unknown_4:@'
	unknown_5:@ 
	unknown_6: '
	unknown_7: 
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????<*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_37979272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????

'
_user_specified_namedense_1_input
?
?
E__inference_conv3d_5_layer_call_and_return_conditional_losses_3797679

inputs<
conv3d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@ *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

 *
paddingVALID*
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
:?????????

 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????

 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????

 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:????????? @
 
_user_specified_nameinputs
?
?
*__inference_conv3d_6_layer_call_fn_3798721

inputs%
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_6_layer_call_and_return_conditional_losses_37977612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????< : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????< 
 
_user_specified_nameinputs
?-
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3798009
dense_1_input"
dense_1_3797978:	
?H
dense_1_3797980:	?H0
conv3d_3_3797984:??
conv3d_3_3797986:	?/
conv3d_4_3797990:?@
conv3d_4_3797992:@.
conv3d_5_3797996:@ 
conv3d_5_3797998: .
conv3d_6_3798002: 
conv3d_6_3798004:
identity?? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall? conv3d_6/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCalldense_1_inputdense_1_3797978dense_1_3797980*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_37975392!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_37975602
reshape/PartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv3d_3_3797984conv3d_3_3797986*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_3_layer_call_and_return_conditional_losses_37975732"
 conv3d_3/StatefulPartitionedCall?
up_sampling3d/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_37976062
up_sampling3d/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall&up_sampling3d/PartitionedCall:output:0conv3d_4_3797990conv3d_4_3797992*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_4_layer_call_and_return_conditional_losses_37976192"
 conv3d_4/StatefulPartitionedCall?
up_sampling3d_1/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:????????? @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_37976662!
up_sampling3d_1/PartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_1/PartitionedCall:output:0conv3d_5_3797996conv3d_5_3797998*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_5_layer_call_and_return_conditional_losses_37976792"
 conv3d_5/StatefulPartitionedCall?
up_sampling3d_2/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????< * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_37977482!
up_sampling3d_2/PartitionedCall?
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_2/PartitionedCall:output:0conv3d_6_3798002conv3d_6_3798004*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_6_layer_call_and_return_conditional_losses_37977612"
 conv3d_6/StatefulPartitionedCall?
cropping3d/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_cropping3d_layer_call_and_return_conditional_losses_37975152
cropping3d/PartitionedCall?
IdentityIdentity#cropping3d/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
: : : : : : : : : : 2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:V R
'
_output_shapes
:?????????

'
_user_specified_namedense_1_input
?

?
D__inference_dense_1_layer_call_and_return_conditional_losses_3798486

inputs1
matmul_readvariableop_resource:	
?H.
biasadd_readvariableop_resource:	?H
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
?H*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????H2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????H2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????H2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
f
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_3797606

inputs
identityd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*
	num_split2
split\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2
concath
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*t
_output_shapesb
`:??????????:??????????:??????????*
	num_split2	
split_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2concat_1/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2

concat_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*t
_output_shapesb
`:??????????:??????????:??????????*
	num_split2	
split_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis?
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2concat_2/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2

concat_2r
IdentityIdentityconcat_2:output:0*
T0*4
_output_shapes"
 :??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?
K
/__inference_up_sampling3d_layer_call_fn_3798531

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_37976062
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
??
?

"__inference__wrapped_model_3797506
dense_1_inputF
3sequential_1_dense_1_matmul_readvariableop_resource:	
?HC
4sequential_1_dense_1_biasadd_readvariableop_resource:	?HT
4sequential_1_conv3d_3_conv3d_readvariableop_resource:??D
5sequential_1_conv3d_3_biasadd_readvariableop_resource:	?S
4sequential_1_conv3d_4_conv3d_readvariableop_resource:?@C
5sequential_1_conv3d_4_biasadd_readvariableop_resource:@R
4sequential_1_conv3d_5_conv3d_readvariableop_resource:@ C
5sequential_1_conv3d_5_biasadd_readvariableop_resource: R
4sequential_1_conv3d_6_conv3d_readvariableop_resource: C
5sequential_1_conv3d_6_biasadd_readvariableop_resource:
identity??,sequential_1/conv3d_3/BiasAdd/ReadVariableOp?+sequential_1/conv3d_3/Conv3D/ReadVariableOp?,sequential_1/conv3d_4/BiasAdd/ReadVariableOp?+sequential_1/conv3d_4/Conv3D/ReadVariableOp?,sequential_1/conv3d_5/BiasAdd/ReadVariableOp?+sequential_1/conv3d_5/Conv3D/ReadVariableOp?,sequential_1/conv3d_6/BiasAdd/ReadVariableOp?+sequential_1/conv3d_6/Conv3D/ReadVariableOp?+sequential_1/dense_1/BiasAdd/ReadVariableOp?*sequential_1/dense_1/MatMul/ReadVariableOp?
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	
?H*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOp?
sequential_1/dense_1/MatMulMatMuldense_1_input2sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????H2
sequential_1/dense_1/MatMul?
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?H*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOp?
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????H2
sequential_1/dense_1/BiasAdd?
sequential_1/dense_1/ReluRelu%sequential_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????H2
sequential_1/dense_1/Relu?
sequential_1/reshape/ShapeShape'sequential_1/dense_1/Relu:activations:0*
T0*
_output_shapes
:2
sequential_1/reshape/Shape?
(sequential_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_1/reshape/strided_slice/stack?
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_1?
*sequential_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_1/reshape/strided_slice/stack_2?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_1/reshape/strided_slice?
$sequential_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/1?
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/2?
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/reshape/Reshape/shape/3?
$sequential_1/reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_1/reshape/Reshape/shape/4?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0-sequential_1/reshape/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/reshape/Reshape/shape?
sequential_1/reshape/ReshapeReshape'sequential_1/dense_1/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*
T0*4
_output_shapes"
 :??????????2
sequential_1/reshape/Reshape?
+sequential_1/conv3d_3/Conv3D/ReadVariableOpReadVariableOp4sequential_1_conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02-
+sequential_1/conv3d_3/Conv3D/ReadVariableOp?
sequential_1/conv3d_3/Conv3DConv3D%sequential_1/reshape/Reshape:output:03sequential_1/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
2
sequential_1/conv3d_3/Conv3D?
,sequential_1/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv3d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_1/conv3d_3/BiasAdd/ReadVariableOp?
sequential_1/conv3d_3/BiasAddBiasAdd%sequential_1/conv3d_3/Conv3D:output:04sequential_1/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????2
sequential_1/conv3d_3/BiasAdd?
sequential_1/conv3d_3/ReluRelu&sequential_1/conv3d_3/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
sequential_1/conv3d_3/Relu?
*sequential_1/up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*sequential_1/up_sampling3d/split/split_dim?
 sequential_1/up_sampling3d/splitSplit3sequential_1/up_sampling3d/split/split_dim:output:0(sequential_1/conv3d_3/Relu:activations:0*
T0*?
_output_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*
	num_split2"
 sequential_1/up_sampling3d/split?
&sequential_1/up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_1/up_sampling3d/concat/axis?
!sequential_1/up_sampling3d/concatConcatV2)sequential_1/up_sampling3d/split:output:0)sequential_1/up_sampling3d/split:output:0)sequential_1/up_sampling3d/split:output:1)sequential_1/up_sampling3d/split:output:1)sequential_1/up_sampling3d/split:output:2)sequential_1/up_sampling3d/split:output:2)sequential_1/up_sampling3d/split:output:3)sequential_1/up_sampling3d/split:output:3)sequential_1/up_sampling3d/split:output:4)sequential_1/up_sampling3d/split:output:4)sequential_1/up_sampling3d/split:output:5)sequential_1/up_sampling3d/split:output:5)sequential_1/up_sampling3d/split:output:6)sequential_1/up_sampling3d/split:output:6)sequential_1/up_sampling3d/split:output:7)sequential_1/up_sampling3d/split:output:7/sequential_1/up_sampling3d/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2#
!sequential_1/up_sampling3d/concat?
,sequential_1/up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_1/up_sampling3d/split_1/split_dim?
"sequential_1/up_sampling3d/split_1Split5sequential_1/up_sampling3d/split_1/split_dim:output:0*sequential_1/up_sampling3d/concat:output:0*
T0*t
_output_shapesb
`:??????????:??????????:??????????*
	num_split2$
"sequential_1/up_sampling3d/split_1?
(sequential_1/up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_1/up_sampling3d/concat_1/axis?
#sequential_1/up_sampling3d/concat_1ConcatV2+sequential_1/up_sampling3d/split_1:output:0+sequential_1/up_sampling3d/split_1:output:0+sequential_1/up_sampling3d/split_1:output:1+sequential_1/up_sampling3d/split_1:output:1+sequential_1/up_sampling3d/split_1:output:2+sequential_1/up_sampling3d/split_1:output:21sequential_1/up_sampling3d/concat_1/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2%
#sequential_1/up_sampling3d/concat_1?
,sequential_1/up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_1/up_sampling3d/split_2/split_dim?
"sequential_1/up_sampling3d/split_2Split5sequential_1/up_sampling3d/split_2/split_dim:output:0,sequential_1/up_sampling3d/concat_1:output:0*
T0*t
_output_shapesb
`:??????????:??????????:??????????*
	num_split2$
"sequential_1/up_sampling3d/split_2?
(sequential_1/up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_1/up_sampling3d/concat_2/axis?
#sequential_1/up_sampling3d/concat_2ConcatV2+sequential_1/up_sampling3d/split_2:output:0+sequential_1/up_sampling3d/split_2:output:0+sequential_1/up_sampling3d/split_2:output:1+sequential_1/up_sampling3d/split_2:output:1+sequential_1/up_sampling3d/split_2:output:2+sequential_1/up_sampling3d/split_2:output:21sequential_1/up_sampling3d/concat_2/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2%
#sequential_1/up_sampling3d/concat_2?
+sequential_1/conv3d_4/Conv3D/ReadVariableOpReadVariableOp4sequential_1_conv3d_4_conv3d_readvariableop_resource*+
_output_shapes
:?@*
dtype02-
+sequential_1/conv3d_4/Conv3D/ReadVariableOp?
sequential_1/conv3d_4/Conv3DConv3D,sequential_1/up_sampling3d/concat_2:output:03sequential_1/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
2
sequential_1/conv3d_4/Conv3D?
,sequential_1/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_1/conv3d_4/BiasAdd/ReadVariableOp?
sequential_1/conv3d_4/BiasAddBiasAdd%sequential_1/conv3d_4/Conv3D:output:04sequential_1/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@2
sequential_1/conv3d_4/BiasAdd?
sequential_1/conv3d_4/ReluRelu&sequential_1/conv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@2
sequential_1/conv3d_4/Relu?
,sequential_1/up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_1/up_sampling3d_1/split/split_dim?
"sequential_1/up_sampling3d_1/splitSplit5sequential_1/up_sampling3d_1/split/split_dim:output:0(sequential_1/conv3d_4/Relu:activations:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2$
"sequential_1/up_sampling3d_1/split?
(sequential_1/up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_1/up_sampling3d_1/concat/axis?
#sequential_1/up_sampling3d_1/concatConcatV2+sequential_1/up_sampling3d_1/split:output:0+sequential_1/up_sampling3d_1/split:output:0+sequential_1/up_sampling3d_1/split:output:1+sequential_1/up_sampling3d_1/split:output:1+sequential_1/up_sampling3d_1/split:output:2+sequential_1/up_sampling3d_1/split:output:2+sequential_1/up_sampling3d_1/split:output:3+sequential_1/up_sampling3d_1/split:output:3+sequential_1/up_sampling3d_1/split:output:4+sequential_1/up_sampling3d_1/split:output:4+sequential_1/up_sampling3d_1/split:output:5+sequential_1/up_sampling3d_1/split:output:5+sequential_1/up_sampling3d_1/split:output:6+sequential_1/up_sampling3d_1/split:output:6+sequential_1/up_sampling3d_1/split:output:7+sequential_1/up_sampling3d_1/split:output:7+sequential_1/up_sampling3d_1/split:output:8+sequential_1/up_sampling3d_1/split:output:8+sequential_1/up_sampling3d_1/split:output:9+sequential_1/up_sampling3d_1/split:output:9,sequential_1/up_sampling3d_1/split:output:10,sequential_1/up_sampling3d_1/split:output:10,sequential_1/up_sampling3d_1/split:output:11,sequential_1/up_sampling3d_1/split:output:11,sequential_1/up_sampling3d_1/split:output:12,sequential_1/up_sampling3d_1/split:output:12,sequential_1/up_sampling3d_1/split:output:13,sequential_1/up_sampling3d_1/split:output:13,sequential_1/up_sampling3d_1/split:output:14,sequential_1/up_sampling3d_1/split:output:14,sequential_1/up_sampling3d_1/split:output:15,sequential_1/up_sampling3d_1/split:output:151sequential_1/up_sampling3d_1/concat/axis:output:0*
N *
T0*3
_output_shapes!
:????????? @2%
#sequential_1/up_sampling3d_1/concat?
.sequential_1/up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.sequential_1/up_sampling3d_1/split_1/split_dim?
$sequential_1/up_sampling3d_1/split_1Split7sequential_1/up_sampling3d_1/split_1/split_dim:output:0,sequential_1/up_sampling3d_1/concat:output:0*
T0*?
_output_shapes?
?:????????? @:????????? @:????????? @:????????? @:????????? @:????????? @*
	num_split2&
$sequential_1/up_sampling3d_1/split_1?
*sequential_1/up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*sequential_1/up_sampling3d_1/concat_1/axis?
%sequential_1/up_sampling3d_1/concat_1ConcatV2-sequential_1/up_sampling3d_1/split_1:output:0-sequential_1/up_sampling3d_1/split_1:output:0-sequential_1/up_sampling3d_1/split_1:output:1-sequential_1/up_sampling3d_1/split_1:output:1-sequential_1/up_sampling3d_1/split_1:output:2-sequential_1/up_sampling3d_1/split_1:output:2-sequential_1/up_sampling3d_1/split_1:output:3-sequential_1/up_sampling3d_1/split_1:output:3-sequential_1/up_sampling3d_1/split_1:output:4-sequential_1/up_sampling3d_1/split_1:output:4-sequential_1/up_sampling3d_1/split_1:output:5-sequential_1/up_sampling3d_1/split_1:output:53sequential_1/up_sampling3d_1/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:????????? @2'
%sequential_1/up_sampling3d_1/concat_1?
.sequential_1/up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.sequential_1/up_sampling3d_1/split_2/split_dim?
$sequential_1/up_sampling3d_1/split_2Split7sequential_1/up_sampling3d_1/split_2/split_dim:output:0.sequential_1/up_sampling3d_1/concat_1:output:0*
T0*?
_output_shapes?
?:????????? @:????????? @:????????? @:????????? @:????????? @:????????? @*
	num_split2&
$sequential_1/up_sampling3d_1/split_2?
*sequential_1/up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*sequential_1/up_sampling3d_1/concat_2/axis?
%sequential_1/up_sampling3d_1/concat_2ConcatV2-sequential_1/up_sampling3d_1/split_2:output:0-sequential_1/up_sampling3d_1/split_2:output:0-sequential_1/up_sampling3d_1/split_2:output:1-sequential_1/up_sampling3d_1/split_2:output:1-sequential_1/up_sampling3d_1/split_2:output:2-sequential_1/up_sampling3d_1/split_2:output:2-sequential_1/up_sampling3d_1/split_2:output:3-sequential_1/up_sampling3d_1/split_2:output:3-sequential_1/up_sampling3d_1/split_2:output:4-sequential_1/up_sampling3d_1/split_2:output:4-sequential_1/up_sampling3d_1/split_2:output:5-sequential_1/up_sampling3d_1/split_2:output:53sequential_1/up_sampling3d_1/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:????????? @2'
%sequential_1/up_sampling3d_1/concat_2?
+sequential_1/conv3d_5/Conv3D/ReadVariableOpReadVariableOp4sequential_1_conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:@ *
dtype02-
+sequential_1/conv3d_5/Conv3D/ReadVariableOp?
sequential_1/conv3d_5/Conv3DConv3D.sequential_1/up_sampling3d_1/concat_2:output:03sequential_1/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

 *
paddingVALID*
strides	
2
sequential_1/conv3d_5/Conv3D?
,sequential_1/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv3d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/conv3d_5/BiasAdd/ReadVariableOp?
sequential_1/conv3d_5/BiasAddBiasAdd%sequential_1/conv3d_5/Conv3D:output:04sequential_1/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

 2
sequential_1/conv3d_5/BiasAdd?
sequential_1/conv3d_5/ReluRelu&sequential_1/conv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????

 2
sequential_1/conv3d_5/Relu?
,sequential_1/up_sampling3d_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_1/up_sampling3d_2/split/split_dim?	
"sequential_1/up_sampling3d_2/splitSplit5sequential_1/up_sampling3d_2/split/split_dim:output:0(sequential_1/conv3d_5/Relu:activations:0*
T0*?
_output_shapes?
?:?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 :?????????

 *
	num_split2$
"sequential_1/up_sampling3d_2/split?
(sequential_1/up_sampling3d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_1/up_sampling3d_2/concat/axis?
#sequential_1/up_sampling3d_2/concatConcatV2+sequential_1/up_sampling3d_2/split:output:0+sequential_1/up_sampling3d_2/split:output:0+sequential_1/up_sampling3d_2/split:output:1+sequential_1/up_sampling3d_2/split:output:1+sequential_1/up_sampling3d_2/split:output:2+sequential_1/up_sampling3d_2/split:output:2+sequential_1/up_sampling3d_2/split:output:3+sequential_1/up_sampling3d_2/split:output:3+sequential_1/up_sampling3d_2/split:output:4+sequential_1/up_sampling3d_2/split:output:4+sequential_1/up_sampling3d_2/split:output:5+sequential_1/up_sampling3d_2/split:output:5+sequential_1/up_sampling3d_2/split:output:6+sequential_1/up_sampling3d_2/split:output:6+sequential_1/up_sampling3d_2/split:output:7+sequential_1/up_sampling3d_2/split:output:7+sequential_1/up_sampling3d_2/split:output:8+sequential_1/up_sampling3d_2/split:output:8+sequential_1/up_sampling3d_2/split:output:9+sequential_1/up_sampling3d_2/split:output:9,sequential_1/up_sampling3d_2/split:output:10,sequential_1/up_sampling3d_2/split:output:10,sequential_1/up_sampling3d_2/split:output:11,sequential_1/up_sampling3d_2/split:output:11,sequential_1/up_sampling3d_2/split:output:12,sequential_1/up_sampling3d_2/split:output:12,sequential_1/up_sampling3d_2/split:output:13,sequential_1/up_sampling3d_2/split:output:13,sequential_1/up_sampling3d_2/split:output:14,sequential_1/up_sampling3d_2/split:output:14,sequential_1/up_sampling3d_2/split:output:15,sequential_1/up_sampling3d_2/split:output:15,sequential_1/up_sampling3d_2/split:output:16,sequential_1/up_sampling3d_2/split:output:16,sequential_1/up_sampling3d_2/split:output:17,sequential_1/up_sampling3d_2/split:output:17,sequential_1/up_sampling3d_2/split:output:18,sequential_1/up_sampling3d_2/split:output:18,sequential_1/up_sampling3d_2/split:output:19,sequential_1/up_sampling3d_2/split:output:19,sequential_1/up_sampling3d_2/split:output:20,sequential_1/up_sampling3d_2/split:output:20,sequential_1/up_sampling3d_2/split:output:21,sequential_1/up_sampling3d_2/split:output:21,sequential_1/up_sampling3d_2/split:output:22,sequential_1/up_sampling3d_2/split:output:22,sequential_1/up_sampling3d_2/split:output:23,sequential_1/up_sampling3d_2/split:output:23,sequential_1/up_sampling3d_2/split:output:24,sequential_1/up_sampling3d_2/split:output:24,sequential_1/up_sampling3d_2/split:output:25,sequential_1/up_sampling3d_2/split:output:25,sequential_1/up_sampling3d_2/split:output:26,sequential_1/up_sampling3d_2/split:output:26,sequential_1/up_sampling3d_2/split:output:27,sequential_1/up_sampling3d_2/split:output:27,sequential_1/up_sampling3d_2/split:output:28,sequential_1/up_sampling3d_2/split:output:28,sequential_1/up_sampling3d_2/split:output:29,sequential_1/up_sampling3d_2/split:output:291sequential_1/up_sampling3d_2/concat/axis:output:0*
N<*
T0*3
_output_shapes!
:?????????<

 2%
#sequential_1/up_sampling3d_2/concat?
.sequential_1/up_sampling3d_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.sequential_1/up_sampling3d_2/split_1/split_dim?
$sequential_1/up_sampling3d_2/split_1Split7sequential_1/up_sampling3d_2/split_1/split_dim:output:0,sequential_1/up_sampling3d_2/concat:output:0*
T0*?
_output_shapes?
?:?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 :?????????<
 *
	num_split
2&
$sequential_1/up_sampling3d_2/split_1?
*sequential_1/up_sampling3d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*sequential_1/up_sampling3d_2/concat_1/axis?	
%sequential_1/up_sampling3d_2/concat_1ConcatV2-sequential_1/up_sampling3d_2/split_1:output:0-sequential_1/up_sampling3d_2/split_1:output:0-sequential_1/up_sampling3d_2/split_1:output:1-sequential_1/up_sampling3d_2/split_1:output:1-sequential_1/up_sampling3d_2/split_1:output:2-sequential_1/up_sampling3d_2/split_1:output:2-sequential_1/up_sampling3d_2/split_1:output:3-sequential_1/up_sampling3d_2/split_1:output:3-sequential_1/up_sampling3d_2/split_1:output:4-sequential_1/up_sampling3d_2/split_1:output:4-sequential_1/up_sampling3d_2/split_1:output:5-sequential_1/up_sampling3d_2/split_1:output:5-sequential_1/up_sampling3d_2/split_1:output:6-sequential_1/up_sampling3d_2/split_1:output:6-sequential_1/up_sampling3d_2/split_1:output:7-sequential_1/up_sampling3d_2/split_1:output:7-sequential_1/up_sampling3d_2/split_1:output:8-sequential_1/up_sampling3d_2/split_1:output:8-sequential_1/up_sampling3d_2/split_1:output:9-sequential_1/up_sampling3d_2/split_1:output:93sequential_1/up_sampling3d_2/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:?????????<
 2'
%sequential_1/up_sampling3d_2/concat_1?
.sequential_1/up_sampling3d_2/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.sequential_1/up_sampling3d_2/split_2/split_dim?
$sequential_1/up_sampling3d_2/split_2Split7sequential_1/up_sampling3d_2/split_2/split_dim:output:0.sequential_1/up_sampling3d_2/concat_1:output:0*
T0*?
_output_shapes?
?:?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< :?????????< *
	num_split
2&
$sequential_1/up_sampling3d_2/split_2?
*sequential_1/up_sampling3d_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*sequential_1/up_sampling3d_2/concat_2/axis?	
%sequential_1/up_sampling3d_2/concat_2ConcatV2-sequential_1/up_sampling3d_2/split_2:output:0-sequential_1/up_sampling3d_2/split_2:output:0-sequential_1/up_sampling3d_2/split_2:output:1-sequential_1/up_sampling3d_2/split_2:output:1-sequential_1/up_sampling3d_2/split_2:output:2-sequential_1/up_sampling3d_2/split_2:output:2-sequential_1/up_sampling3d_2/split_2:output:3-sequential_1/up_sampling3d_2/split_2:output:3-sequential_1/up_sampling3d_2/split_2:output:4-sequential_1/up_sampling3d_2/split_2:output:4-sequential_1/up_sampling3d_2/split_2:output:5-sequential_1/up_sampling3d_2/split_2:output:5-sequential_1/up_sampling3d_2/split_2:output:6-sequential_1/up_sampling3d_2/split_2:output:6-sequential_1/up_sampling3d_2/split_2:output:7-sequential_1/up_sampling3d_2/split_2:output:7-sequential_1/up_sampling3d_2/split_2:output:8-sequential_1/up_sampling3d_2/split_2:output:8-sequential_1/up_sampling3d_2/split_2:output:9-sequential_1/up_sampling3d_2/split_2:output:93sequential_1/up_sampling3d_2/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:?????????< 2'
%sequential_1/up_sampling3d_2/concat_2?
+sequential_1/conv3d_6/Conv3D/ReadVariableOpReadVariableOp4sequential_1_conv3d_6_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02-
+sequential_1/conv3d_6/Conv3D/ReadVariableOp?
sequential_1/conv3d_6/Conv3DConv3D.sequential_1/up_sampling3d_2/concat_2:output:03sequential_1/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????<*
paddingSAME*
strides	
2
sequential_1/conv3d_6/Conv3D?
,sequential_1/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/conv3d_6/BiasAdd/ReadVariableOp?
sequential_1/conv3d_6/BiasAddBiasAdd%sequential_1/conv3d_6/Conv3D:output:04sequential_1/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????<2
sequential_1/conv3d_6/BiasAdd?
sequential_1/conv3d_6/SigmoidSigmoid&sequential_1/conv3d_6/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????<2
sequential_1/conv3d_6/Sigmoid?
+sequential_1/cropping3d/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    2-
+sequential_1/cropping3d/strided_slice/stack?
-sequential_1/cropping3d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    2/
-sequential_1/cropping3d/strided_slice/stack_1?
-sequential_1/cropping3d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               2/
-sequential_1/cropping3d/strided_slice/stack_2?
%sequential_1/cropping3d/strided_sliceStridedSlice!sequential_1/conv3d_6/Sigmoid:y:04sequential_1/cropping3d/strided_slice/stack:output:06sequential_1/cropping3d/strided_slice/stack_1:output:06sequential_1/cropping3d/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:?????????<*

begin_mask*
end_mask2'
%sequential_1/cropping3d/strided_slice?
IdentityIdentity.sequential_1/cropping3d/strided_slice:output:0-^sequential_1/conv3d_3/BiasAdd/ReadVariableOp,^sequential_1/conv3d_3/Conv3D/ReadVariableOp-^sequential_1/conv3d_4/BiasAdd/ReadVariableOp,^sequential_1/conv3d_4/Conv3D/ReadVariableOp-^sequential_1/conv3d_5/BiasAdd/ReadVariableOp,^sequential_1/conv3d_5/Conv3D/ReadVariableOp-^sequential_1/conv3d_6/BiasAdd/ReadVariableOp,^sequential_1/conv3d_6/Conv3D/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*
T0*3
_output_shapes!
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
: : : : : : : : : : 2\
,sequential_1/conv3d_3/BiasAdd/ReadVariableOp,sequential_1/conv3d_3/BiasAdd/ReadVariableOp2Z
+sequential_1/conv3d_3/Conv3D/ReadVariableOp+sequential_1/conv3d_3/Conv3D/ReadVariableOp2\
,sequential_1/conv3d_4/BiasAdd/ReadVariableOp,sequential_1/conv3d_4/BiasAdd/ReadVariableOp2Z
+sequential_1/conv3d_4/Conv3D/ReadVariableOp+sequential_1/conv3d_4/Conv3D/ReadVariableOp2\
,sequential_1/conv3d_5/BiasAdd/ReadVariableOp,sequential_1/conv3d_5/BiasAdd/ReadVariableOp2Z
+sequential_1/conv3d_5/Conv3D/ReadVariableOp+sequential_1/conv3d_5/Conv3D/ReadVariableOp2\
,sequential_1/conv3d_6/BiasAdd/ReadVariableOp,sequential_1/conv3d_6/BiasAdd/ReadVariableOp2Z
+sequential_1/conv3d_6/Conv3D/ReadVariableOp+sequential_1/conv3d_6/Conv3D/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp:V R
'
_output_shapes
:?????????

'
_user_specified_namedense_1_input
?

?
D__inference_dense_1_layer_call_and_return_conditional_losses_3797539

inputs1
matmul_readvariableop_resource:	
?H.
biasadd_readvariableop_resource:	?H
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
?H*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????H2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????H2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????H2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?-
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3797769

inputs"
dense_1_3797540:	
?H
dense_1_3797542:	?H0
conv3d_3_3797574:??
conv3d_3_3797576:	?/
conv3d_4_3797620:?@
conv3d_4_3797622:@.
conv3d_5_3797680:@ 
conv3d_5_3797682: .
conv3d_6_3797762: 
conv3d_6_3797764:
identity?? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall? conv3d_6/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_3797540dense_1_3797542*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_37975392!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_37975602
reshape/PartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv3d_3_3797574conv3d_3_3797576*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_3_layer_call_and_return_conditional_losses_37975732"
 conv3d_3/StatefulPartitionedCall?
up_sampling3d/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_37976062
up_sampling3d/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall&up_sampling3d/PartitionedCall:output:0conv3d_4_3797620conv3d_4_3797622*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_4_layer_call_and_return_conditional_losses_37976192"
 conv3d_4/StatefulPartitionedCall?
up_sampling3d_1/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:????????? @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_37976662!
up_sampling3d_1/PartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_1/PartitionedCall:output:0conv3d_5_3797680conv3d_5_3797682*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_5_layer_call_and_return_conditional_losses_37976792"
 conv3d_5/StatefulPartitionedCall?
up_sampling3d_2/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????< * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_37977482!
up_sampling3d_2/PartitionedCall?
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_2/PartitionedCall:output:0conv3d_6_3797762conv3d_6_3797764*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_6_layer_call_and_return_conditional_losses_37977612"
 conv3d_6/StatefulPartitionedCall?
cropping3d/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_cropping3d_layer_call_and_return_conditional_losses_37975152
cropping3d/PartitionedCall?
IdentityIdentity#cropping3d/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
: : : : : : : : : : 2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
)__inference_dense_1_layer_call_fn_3798475

inputs
unknown:	
?H
	unknown_0:	?H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_37975392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
`
D__inference_reshape_layer_call_and_return_conditional_losses_3798506

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
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3e
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/4?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape|
ReshapeReshapeinputsReshape/shape:output:0*
T0*4
_output_shapes"
 :??????????2	
Reshapeq
IdentityIdentityReshape:output:0*
T0*4
_output_shapes"
 :??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????H:P L
(
_output_shapes
:??????????H
 
_user_specified_nameinputs
?

?
.__inference_sequential_1_layer_call_fn_3797792
dense_1_input
unknown:	
?H
	unknown_0:	?H)
	unknown_1:??
	unknown_2:	?(
	unknown_3:?@
	unknown_4:@'
	unknown_5:@ 
	unknown_6: '
	unknown_7: 
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????<*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_37977692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????

'
_user_specified_namedense_1_input
?
?
E__inference_conv3d_4_layer_call_and_return_conditional_losses_3798578

inputs=
conv3d_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:?@*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
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
:?????????@2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?
?
*__inference_conv3d_5_layer_call_fn_3798633

inputs%
unknown:@ 
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
:?????????

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_5_layer_call_and_return_conditional_losses_37976792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????

 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? @: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:????????? @
 
_user_specified_nameinputs
?
?
E__inference_conv3d_5_layer_call_and_return_conditional_losses_3798644

inputs<
conv3d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@ *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????

 *
paddingVALID*
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
:?????????

 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????

 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????

 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:????????? @
 
_user_specified_nameinputs
?
H
,__inference_cropping3d_layer_call_fn_3797521

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
GPU2*0J 8? *P
fKRI
G__inference_cropping3d_layer_call_and_return_conditional_losses_37975152
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
?.
?
#__inference__traced_restore_3798825
file_prefix2
assignvariableop_dense_1_kernel:	
?H.
assignvariableop_1_dense_1_bias:	?HB
"assignvariableop_2_conv3d_3_kernel:??/
 assignvariableop_3_conv3d_3_bias:	?A
"assignvariableop_4_conv3d_4_kernel:?@.
 assignvariableop_5_conv3d_4_bias:@@
"assignvariableop_6_conv3d_5_kernel:@ .
 assignvariableop_7_conv3d_5_bias: @
"assignvariableop_8_conv3d_6_kernel: .
 assignvariableop_9_conv3d_6_bias:
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv3d_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv3d_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv3d_4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv3d_4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv3d_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv3d_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv3d_6_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv3d_6_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10?
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
.__inference_sequential_1_layer_call_fn_3798120

inputs
unknown:	
?H
	unknown_0:	?H)
	unknown_1:??
	unknown_2:	?(
	unknown_3:?@
	unknown_4:@'
	unknown_5:@ 
	unknown_6: '
	unknown_7: 
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????<*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_37979272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
E__inference_conv3d_3_layer_call_and_return_conditional_losses_3798526

inputs>
conv3d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
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
 :??????????2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*4
_output_shapes"
 :??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?
?
E__inference_conv3d_6_layer_call_and_return_conditional_losses_3797761

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????<*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????<2	
BiasAddm
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????<2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????< : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????< 
 
_user_specified_nameinputs
?	
c
G__inference_cropping3d_layer_call_and_return_conditional_losses_3797515

inputs
identity?
strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*

begin_mask*
end_mask2
strided_slice?
IdentityIdentitystrided_slice:output:0*
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
?-
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3797927

inputs"
dense_1_3797896:	
?H
dense_1_3797898:	?H0
conv3d_3_3797902:??
conv3d_3_3797904:	?/
conv3d_4_3797908:?@
conv3d_4_3797910:@.
conv3d_5_3797914:@ 
conv3d_5_3797916: .
conv3d_6_3797920: 
conv3d_6_3797922:
identity?? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall? conv3d_6/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_3797896dense_1_3797898*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_37975392!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_37975602
reshape/PartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv3d_3_3797902conv3d_3_3797904*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_3_layer_call_and_return_conditional_losses_37975732"
 conv3d_3/StatefulPartitionedCall?
up_sampling3d/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_37976062
up_sampling3d/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall&up_sampling3d/PartitionedCall:output:0conv3d_4_3797908conv3d_4_3797910*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_4_layer_call_and_return_conditional_losses_37976192"
 conv3d_4/StatefulPartitionedCall?
up_sampling3d_1/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:????????? @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_37976662!
up_sampling3d_1/PartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_1/PartitionedCall:output:0conv3d_5_3797914conv3d_5_3797916*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_5_layer_call_and_return_conditional_losses_37976792"
 conv3d_5/StatefulPartitionedCall?
up_sampling3d_2/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????< * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_37977482!
up_sampling3d_2/PartitionedCall?
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling3d_2/PartitionedCall:output:0conv3d_6_3797920conv3d_6_3797922*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_6_layer_call_and_return_conditional_losses_37977612"
 conv3d_6/StatefulPartitionedCall?
cropping3d/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_cropping3d_layer_call_and_return_conditional_losses_37975152
cropping3d/PartitionedCall?
IdentityIdentity#cropping3d/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
: : : : : : : : : : 2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?

?
%__inference_signature_wrapper_3798070
dense_1_input
unknown:	
?H
	unknown_0:	?H)
	unknown_1:??
	unknown_2:	?(
	unknown_3:?@
	unknown_4:@'
	unknown_5:@ 
	unknown_6: '
	unknown_7: 
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????<*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_37975062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????

'
_user_specified_namedense_1_input
?
M
1__inference_up_sampling3d_1_layer_call_fn_3798583

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:????????? @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_37976662
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:????????? @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
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
dense_1_input6
serving_default_dense_1_input:0?????????
J

cropping3d<
StatefulPartitionedCall:0?????????<tensorflow/serving/predict:??
?W
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9

signatures
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?T
_tf_keras_sequential?S{"name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_1_input"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "units": 9216, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 3, 3, 128]}}}, {"class_name": "Conv3D", "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "UpSampling3D", "config": {"name": "up_sampling3d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv3D", "config": {"name": "conv3d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "UpSampling3D", "config": {"name": "up_sampling3d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv3D", "config": {"name": "conv3d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "UpSampling3D", "config": {"name": "up_sampling3d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv3D", "config": {"name": "conv3d_6", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Cropping3D", "config": {"name": "cropping3d", "trainable": true, "dtype": "float32", "cropping": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 0]}]}, "data_format": "channels_last"}}]}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 10]}, "float32", "dense_1_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_1_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "units": 9216, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 3, 3, 128]}}, "shared_object_id": 4}, {"class_name": "Conv3D", "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, {"class_name": "UpSampling3D", "config": {"name": "up_sampling3d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7}, {"class_name": "Conv3D", "config": {"name": "conv3d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9}, {"class_name": "UpSampling3D", "config": {"name": "up_sampling3d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 10}, {"class_name": "Conv3D", "config": {"name": "conv3d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "UpSampling3D", "config": {"name": "up_sampling3d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv3D", "config": {"name": "conv3d_6", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15}, {"class_name": "Cropping3D", "config": {"name": "cropping3d", "trainable": true, "dtype": "float32", "cropping": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 0]}]}, "data_format": "channels_last"}, "shared_object_id": 16}]}}}
?


kernel
bias
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "units": 9216, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 3, 3, 128]}}, "shared_object_id": 4}
?

kernel
bias
#_self_saveable_object_factories
 	variables
!regularization_losses
"trainable_variables
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv3d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 128}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 3, 3, 128]}}
?
#$_self_saveable_object_factories
%	variables
&regularization_losses
'trainable_variables
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "up_sampling3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling3D", "config": {"name": "up_sampling3d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7}
?

)kernel
*bias
#+_self_saveable_object_factories
,	variables
-regularization_losses
.trainable_variables
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv3d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 128}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 6, 6, 128]}}
?
#0_self_saveable_object_factories
1	variables
2regularization_losses
3trainable_variables
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "up_sampling3d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling3D", "config": {"name": "up_sampling3d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 10}
?

5kernel
6bias
#7_self_saveable_object_factories
8	variables
9regularization_losses
:trainable_variables
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv3d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 64}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 12, 12, 64]}}
?
#<_self_saveable_object_factories
=	variables
>regularization_losses
?trainable_variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "up_sampling3d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling3D", "config": {"name": "up_sampling3d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}
?

Akernel
Bbias
#C_self_saveable_object_factories
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv3d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv3D", "config": {"name": "conv3d_6", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 32}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 20, 20, 32]}}
?
#H_self_saveable_object_factories
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "cropping3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Cropping3D", "config": {"name": "cropping3d", "trainable": true, "dtype": "float32", "cropping": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 0]}]}, "data_format": "channels_last"}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 23}}
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
f
0
1
2
3
)4
*5
56
67
A8
B9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
)4
*5
56
67
A8
B9"
trackable_list_wrapper
?
Mlayer_metrics
	variables

Nlayers
Onon_trainable_variables
Pmetrics
regularization_losses
trainable_variables
Qlayer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	
?H2dense_1/kernel
:?H2dense_1/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Rlayer_metrics
	variables

Slayers
Tnon_trainable_variables
Umetrics
regularization_losses
trainable_variables
Vlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
Wlayer_metrics
	variables

Xlayers
Ynon_trainable_variables
Zmetrics
regularization_losses
trainable_variables
[layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2conv3d_3/kernel
:?2conv3d_3/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
\layer_metrics
 	variables

]layers
^non_trainable_variables
_metrics
!regularization_losses
"trainable_variables
`layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
alayer_metrics
%	variables

blayers
cnon_trainable_variables
dmetrics
&regularization_losses
'trainable_variables
elayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,?@2conv3d_4/kernel
:@2conv3d_4/bias
 "
trackable_dict_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
flayer_metrics
,	variables

glayers
hnon_trainable_variables
imetrics
-regularization_losses
.trainable_variables
jlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
klayer_metrics
1	variables

llayers
mnon_trainable_variables
nmetrics
2regularization_losses
3trainable_variables
olayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+@ 2conv3d_5/kernel
: 2conv3d_5/bias
 "
trackable_dict_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
player_metrics
8	variables

qlayers
rnon_trainable_variables
smetrics
9regularization_losses
:trainable_variables
tlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
ulayer_metrics
=	variables

vlayers
wnon_trainable_variables
xmetrics
>regularization_losses
?trainable_variables
ylayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+ 2conv3d_6/kernel
:2conv3d_6/bias
 "
trackable_dict_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
?
zlayer_metrics
D	variables

{layers
|non_trainable_variables
}metrics
Eregularization_losses
Ftrainable_variables
~layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
layer_metrics
I	variables
?layers
?non_trainable_variables
?metrics
Jregularization_losses
Ktrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
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
9"
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
?2?
"__inference__wrapped_model_3797506?
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
dense_1_input?????????

?2?
.__inference_sequential_1_layer_call_fn_3797792
.__inference_sequential_1_layer_call_fn_3798095
.__inference_sequential_1_layer_call_fn_3798120
.__inference_sequential_1_layer_call_fn_3797975?
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
I__inference_sequential_1_layer_call_and_return_conditional_losses_3798293
I__inference_sequential_1_layer_call_and_return_conditional_losses_3798466
I__inference_sequential_1_layer_call_and_return_conditional_losses_3798009
I__inference_sequential_1_layer_call_and_return_conditional_losses_3798043?
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
)__inference_dense_1_layer_call_fn_3798475?
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
D__inference_dense_1_layer_call_and_return_conditional_losses_3798486?
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
)__inference_reshape_layer_call_fn_3798491?
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
D__inference_reshape_layer_call_and_return_conditional_losses_3798506?
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
*__inference_conv3d_3_layer_call_fn_3798515?
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
E__inference_conv3d_3_layer_call_and_return_conditional_losses_3798526?
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
/__inference_up_sampling3d_layer_call_fn_3798531?
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
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_3798558?
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
*__inference_conv3d_4_layer_call_fn_3798567?
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
E__inference_conv3d_4_layer_call_and_return_conditional_losses_3798578?
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
1__inference_up_sampling3d_1_layer_call_fn_3798583?
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
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_3798624?
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
*__inference_conv3d_5_layer_call_fn_3798633?
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
E__inference_conv3d_5_layer_call_and_return_conditional_losses_3798644?
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
1__inference_up_sampling3d_2_layer_call_fn_3798649?
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
L__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_3798712?
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
*__inference_conv3d_6_layer_call_fn_3798721?
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
E__inference_conv3d_6_layer_call_and_return_conditional_losses_3798732?
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
,__inference_cropping3d_layer_call_fn_3797521?
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
G__inference_cropping3d_layer_call_and_return_conditional_losses_3797515?
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
?B?
%__inference_signature_wrapper_3798070dense_1_input"?
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
"__inference__wrapped_model_3797506?
)*56AB6?3
,?)
'?$
dense_1_input?????????

? "C?@
>

cropping3d0?-

cropping3d?????????<?
E__inference_conv3d_3_layer_call_and_return_conditional_losses_3798526v<?9
2?/
-?*
inputs??????????
? "2?/
(?%
0??????????
? ?
*__inference_conv3d_3_layer_call_fn_3798515i<?9
2?/
-?*
inputs??????????
? "%?"???????????
E__inference_conv3d_4_layer_call_and_return_conditional_losses_3798578u)*<?9
2?/
-?*
inputs??????????
? "1?.
'?$
0?????????@
? ?
*__inference_conv3d_4_layer_call_fn_3798567h)*<?9
2?/
-?*
inputs??????????
? "$?!?????????@?
E__inference_conv3d_5_layer_call_and_return_conditional_losses_3798644t56;?8
1?.
,?)
inputs????????? @
? "1?.
'?$
0?????????

 
? ?
*__inference_conv3d_5_layer_call_fn_3798633g56;?8
1?.
,?)
inputs????????? @
? "$?!?????????

 ?
E__inference_conv3d_6_layer_call_and_return_conditional_losses_3798732tAB;?8
1?.
,?)
inputs?????????< 
? "1?.
'?$
0?????????<
? ?
*__inference_conv3d_6_layer_call_fn_3798721gAB;?8
1?.
,?)
inputs?????????< 
? "$?!?????????<?
G__inference_cropping3d_layer_call_and_return_conditional_losses_3797515?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
,__inference_cropping3d_layer_call_fn_3797521?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
D__inference_dense_1_layer_call_and_return_conditional_losses_3798486]/?,
%?"
 ?
inputs?????????

? "&?#
?
0??????????H
? }
)__inference_dense_1_layer_call_fn_3798475P/?,
%?"
 ?
inputs?????????

? "???????????H?
D__inference_reshape_layer_call_and_return_conditional_losses_3798506f0?-
&?#
!?
inputs??????????H
? "2?/
(?%
0??????????
? ?
)__inference_reshape_layer_call_fn_3798491Y0?-
&?#
!?
inputs??????????H
? "%?"???????????
I__inference_sequential_1_layer_call_and_return_conditional_losses_3798009
)*56AB>?;
4?1
'?$
dense_1_input?????????

p 

 
? "1?.
'?$
0?????????<
? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3798043
)*56AB>?;
4?1
'?$
dense_1_input?????????

p

 
? "1?.
'?$
0?????????<
? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3798293x
)*56AB7?4
-?*
 ?
inputs?????????

p 

 
? "1?.
'?$
0?????????<
? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3798466x
)*56AB7?4
-?*
 ?
inputs?????????

p

 
? "1?.
'?$
0?????????<
? ?
.__inference_sequential_1_layer_call_fn_3797792r
)*56AB>?;
4?1
'?$
dense_1_input?????????

p 

 
? "$?!?????????<?
.__inference_sequential_1_layer_call_fn_3797975r
)*56AB>?;
4?1
'?$
dense_1_input?????????

p

 
? "$?!?????????<?
.__inference_sequential_1_layer_call_fn_3798095k
)*56AB7?4
-?*
 ?
inputs?????????

p 

 
? "$?!?????????<?
.__inference_sequential_1_layer_call_fn_3798120k
)*56AB7?4
-?*
 ?
inputs?????????

p

 
? "$?!?????????<?
%__inference_signature_wrapper_3798070?
)*56ABG?D
? 
=?:
8
dense_1_input'?$
dense_1_input?????????
"C?@
>

cropping3d0?-

cropping3d?????????<?
L__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_3798624p;?8
1?.
,?)
inputs?????????@
? "1?.
'?$
0????????? @
? ?
1__inference_up_sampling3d_1_layer_call_fn_3798583c;?8
1?.
,?)
inputs?????????@
? "$?!????????? @?
L__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_3798712p;?8
1?.
,?)
inputs?????????

 
? "1?.
'?$
0?????????< 
? ?
1__inference_up_sampling3d_2_layer_call_fn_3798649c;?8
1?.
,?)
inputs?????????

 
? "$?!?????????< ?
J__inference_up_sampling3d_layer_call_and_return_conditional_losses_3798558r<?9
2?/
-?*
inputs??????????
? "2?/
(?%
0??????????
? ?
/__inference_up_sampling3d_layer_call_fn_3798531e<?9
2?/
-?*
inputs??????????
? "%?"??????????