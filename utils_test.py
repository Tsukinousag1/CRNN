import sys
import unittest
import torch
from torch.autograd import Variable
import collections
import utils

def equal(a,b):
    if isinstance(a,torch.Tensor):
        return a.equal(b)#判断是否相等
    elif isinstance(a,str):
        return a==b
    #迭代器类型
    elif isinstance(a,collections.Iterable):
        res=True
        for (x,y) in zip(a,b):
            res=res & equal(x,y)
        return res
    else:
        return a==b

class utilsTestCase(unittest.TestCase):

    def test_Converter(self):
        encoder=utils.strLabelConverter('abcdefghijklmnopqrstuvwxyz')

        #Encoder
        #trivial mode
        result=encoder.encode('efa')
        target=(torch.IntTensor([5,6,1]),torch.IntTensor([3]))
        self.assertTrue(equal(result,target))

        #batch mode
        result=encoder.encode(['efa','ab'])
        target=(torch.IntTensor([5,6,1,1,2]),torch.IntTensor([3,2]))
        self.assertTrue(equal(result,target))

        #decode
        #trivial mode
        result=encoder.decode(torch.IntTensor([5,6,1]),torch.IntTensor([3]))
        target='efa'
        self.assertTrue(equal(result,target))

        #replicate mode ee_a
        result=encoder.decode(torch.IntTensor([5,5,0,1]),torch.IntTensor([4]))
        target='ea'
        self.assertTrue(equal(result,target))

        #raise AssertionError
        def f():
            result=encoder.encode(
                torch.IntTensor([5,5,0,1]),torch.IntTensor([3]))
            return result
        self.assertTrue(AssertionError,f)

        #batch mode
        result=encoder.decode(
            torch.IntTensor([5,6,1,1,2]),torch.IntTensor([3,2]))
        target=['efa','ab']
        self.assertTrue(equal(result,target))

    def test_OneHot(self):
        v=torch.LongTensor([1,2,1,2,0])
        v_length=torch.LongTensor([2,3])
        v_onehot=utils.oneHot(v,v_length,4)#nc=4,输入通道
        target=torch.FloatTensor([[[0,1,0,0],
                                   [0,0,1,0],
                                   [0,0,0,0]],#1 2 空

                                  [[0,1,0,0],
                                   [0,0,1,0],
                                   [1,0,0,0]]])#1 2 0
        assert target.equal(v_onehot)

    def test_Averager(self):
        #1+2+5+6=14
        #14/4=3.5
        acc=utils.averager()
        acc.add(Variable(torch.Tensor([1,2])))
        acc.add(Variable(torch.Tensor([[5,6]])))
        assert acc.val()==3.5

        acc=utils.averager()
        acc.add(torch.Tensor([1,2]))
        acc.add(torch.Tensor([[5,6]]))
        assert acc.val()==3.5

    def test_AssureRatio(self):
    #"""Ensure imgH<=imgW"""
        img=torch.Tensor([[1],[3]]).view(1,1,2,1)
        img=Variable(img)
    #还是torch.Tensor
        img=utils.assureRatio(img)
        assert torch.Size([1,1,2,2])==img.size()

def _suite():
    suite=unittest.TestSuite()
    suite.addTest(utilsTestCase("test_Converter"))
    suite.addTest(utilsTestCase("test_OneHot"))
    suite.addTest(utilsTestCase("test_Averager"))
    suite.addTest(utilsTestCase("test_AssureRatio"))
    return suite

if __name__=="__main__":
    suite=_suite()
    runner=unittest.TextTestRunner()
    runner.run(suite)

