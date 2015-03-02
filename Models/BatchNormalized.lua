

require 'cudnn'
require 'eladtools'
local model = nn.Sequential() 

-- Convolution Layers

model:add(cudnn.SpatialConvolution(3, 192, 5, 5,1,1,2,2 ))
model:add(cudnn.ReLU(true))
model:add(nn.BatchNormalization())
model:add(nn.SpatialConvolutionMM(192, 160,1,1 ))
model:add(cudnn.ReLU(true))
model:add(nn.BatchNormalization())
model:add(nn.SpatialConvolutionMM(160,96, 1,1 ))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(3, 3,2,2))
model:add(nn.BatchNormalization())
model:add(cudnn.SpatialConvolution(96,192, 5,5,1,1,2,2 ))
model:add(cudnn.ReLU(true))
model:add(nn.BatchNormalization())
model:add(nn.SpatialConvolutionMM(192,192, 1,1 ))
model:add(cudnn.ReLU(true))

model:add(nn.BatchNormalization())
model:add(nn.SpatialConvolutionMM(192,192, 1,1 ))
model:add(cudnn.ReLU(true))
model:add(nn.BatchNormalization())
model:add(cudnn.SpatialMaxPooling(3,3,2,2))
model:add(nn.BatchNormalization())
model:add(cudnn.SpatialConvolution(192,192, 3,3 ,1,1,1,1))
model:add(cudnn.ReLU(true))
model:add(nn.BatchNormalization())
model:add(nn.SpatialConvolutionMM(192,192, 1,1 ))
model:add(cudnn.ReLU(true))
model:add(nn.BatchNormalization())
model:add(nn.SpatialConvolutionMM(192,10, 1,1 ))

model:add(cudnn.SpatialAveragePooling(7,7))
model:add(nn.View(10))
model:add(nn.LogSoftMax())


for i,layer in ipairs(model.modules) do
   if layer.bias then
      layer.bias:fill(0)
      layer.weight:normal(0, 0.05)
   end
end
return model
