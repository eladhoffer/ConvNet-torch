require 'nn'
local backend_name = 'cudnn'
local backend
if backend_name == 'cudnn' then
  require 'cudnn'
  backend = cudnn
else
  backend = nn
end


local model = nn.Sequential()

-- Convolution Layers

model:add(backend.SpatialConvolution(3, 192, 5, 5,1,1,2,2 ))
model:add(backend.ReLU(true))
model:add(backend.SpatialBatchNormalization(192))
model:add(backend.SpatialConvolution(192, 160,1,1 ))
model:add(backend.ReLU(true))
model:add(backend.SpatialBatchNormalization(160))
model:add(backend.SpatialConvolution(160,96, 1,1 ))
model:add(backend.ReLU(true))
model:add(backend.SpatialMaxPooling(3, 3,2,2))
model:add(backend.SpatialBatchNormalization(96))
model:add(nn.Dropout(0.5))
model:add(backend.SpatialConvolution(96,192, 5,5,1,1,2,2 ))
model:add(backend.ReLU(true))
model:add(backend.SpatialBatchNormalization(192))
model:add(backend.SpatialConvolution(192,192, 1,1 ))
model:add(backend.ReLU(true))

model:add(backend.SpatialBatchNormalization(192))
model:add(backend.SpatialConvolution(192,192, 1,1 ))
model:add(backend.ReLU(true))
model:add(backend.SpatialMaxPooling(3,3,2,2))
model:add(backend.SpatialBatchNormalization(192))
model:add(nn.Dropout(0.5))
model:add(backend.SpatialConvolution(192,192, 3,3 ,1,1,1,1))
model:add(backend.ReLU(true))
model:add(backend.SpatialBatchNormalization(192))
model:add(backend.SpatialConvolution(192,192, 1,1 ))
model:add(backend.ReLU(true))
model:add(nn.Dropout(0.5))
model:add(backend.SpatialBatchNormalization(192))
model:add(backend.SpatialConvolution(192,10, 1,1 ))

model:add(backend.SpatialAveragePooling(7,7))
model:add(nn.View(10))
model:add(backend.LogSoftMax())


--for i,layer in ipairs(model.modules) do
--   if layer.bias then
--      layer.bias:fill(0)
--      layer.weight:normal(0, 0.05)
--   end
--end
return model
