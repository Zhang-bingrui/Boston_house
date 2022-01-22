# Boston_house
波士顿房价数据集
目的：通过分析十三个房屋特征与房价的关系，同时建立模型进行房价预测
波士顿房价指标与房价的关系
CRIM：城镇人均犯罪率——负相关占比

ZN：住宅用地所占比例——无单个特征

INDUS：城镇中非住宅用地所占比例——负相关

CHAS：虚拟变量，用于回归分析——无单个特征

NOX：环保指数——无单个特征

RM：每栋住宅的房间数——正相关

AGE：1940年以前建成的自住单位的比例——无单个特征


DIS：距离5个波士顿的就业中心的加权距离——无单个特征

RAD：距离高速公路的便利指数——无单个特征

TAX：每一万美元的不动产税率——无单个特征

PTRATIO：城镇中教师学生比例——无单个特征

B：城镇中黑人的比例——无单个特征

LSTAT：地区中多少房东属于低收入人群——负相关

MEDV：自主房屋房价中位数（标签数据）——房价中位数


通过绘制13个特征单个特征与房价中位数之间的关系，可以看出，单个特征可能与房价呈现明显正相关如RM——每栋住宅的房间数，也可能呈现明显的负相关如CRIM_城镇人均犯罪率、INDUS——城镇中非住宅用地所占比例和LSTAT——地区中多少房东属于低收入人群。
同时，通过随机协方差的分析方式可以看出，每个特征之间存在一定的线性相关因素，因此每个特征都对房价有影响，不可剔除作为不影响因素。

目前通过简单线性回归和随机森林回归的方式，通过比较平均绝对误差的大小，可以看出两个模型的好坏。