
require 'cunn'
require 'cudnn'
local model = nn.Sequential() 
local final_mlpconv_layer = nil

-- Convolution Layers

model:add(cudnn.SpatialConvolution(3, 64, 5, 5 ))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2, 2))
model:add(nn.Dropout(0.5))

model:add(cudnn.SpatialConvolution(64, 128, 3, 3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2, 2))
model:add(nn.Dropout(0.5))

model:add(cudnn.SpatialConvolution(128, 256, 3, 3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2, 2))


--Test New
--

model:add(nn.Dropout(0.5))
model:add(cudnn.SpatialConvolution(256, 128, 2,2))
model:add(cudnn.ReLU(true))
model:add(nn.View(128))
model:add(nn.Dropout(0.5))


model:add(nn.Linear(128,10))
model:add(nn.LogSoftMax())
return model
