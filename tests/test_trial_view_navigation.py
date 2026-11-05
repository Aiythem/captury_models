import sys,unittest,os
from types import SimpleNamespace
import tkinter as tk
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gui_trial_viewer import TkC3DTrialCanvas,JointCentreChainData,open_trial_view_window
from c3d_trial_viewer import C3DMarkerData
@unittest.skipUnless(os.environ.get("RUN_TK_SMOKE") == "1", "Set RUN_TK_SMOKE=1 with desktop access")
class NavigationTests(unittest.TestCase):
 def setUp(self):
  self.root=tk.Tk();self.root.geometry('700x600')
  self.v=TkC3DTrialCanvas(self.root);self.v.pack(fill='both',expand=True)
  points=np.array([[[0.,0.],[100.,100.]],[[0.,0.],[0.,0.]],[[0.,0.],[500.,500.]]])
  self.v.set_marker_layers({'motive':C3DMarkerData(labels=['RFLE','RFME'],points=points,rate=120,unit='mm')})
  chain={'right_thigh':np.array([[0,0,500],[0,0,500]]),'right_shank':np.array([[0,0,0],[0,0,0]])}
  self.v.set_joint_centre_chains(JointCentreChainData(layers={'motive':chain,'biobuddy':chain},edges=(('right_thigh','right_shank'),)))
  self.v.set_visible_cor_layers({'motive','biobuddy'});self.root.update()
 def tearDown(self):self.root.destroy()
 def test_pan_moves_scene_without_rotation(self):
  self.v.camera=np.eye(3);self.v.redraw();old=self.v.camera.copy()
  before=[self.v.coords(i) for i in self.v.find_all() if self.v.type(i)=='oval'][0]
  self.v._on_mouse_down(SimpleNamespace(state=1,x=10,y=20))
  self.v._on_mouse_drag(SimpleNamespace(x=40,y=60));self.v.redraw()
  after=[self.v.coords(i) for i in self.v.find_all() if self.v.type(i)=='oval'][0]
  np.testing.assert_allclose(np.array(after)-before,[30,40,30,40]);np.testing.assert_array_equal(self.v.camera,old)
  self.v.reset_camera();np.testing.assert_array_equal(self.v.pan_pixels,[0,0])
 def test_large_window_retains_chains_and_navigation(self):
  self.v.set_frame(1);self.v._zoom(2)
  win=open_trial_view_window(self.root,self.v);self.root.update();v=win.viewer
  self.assertEqual(v.frame,1);self.assertEqual(v.zoom,2);self.assertEqual(v.visible_cor_layers,{'motive','biobuddy'})
  colors=[v.itemcget(i,'fill') for i in v.find_all() if v.type(i)=='line']
  self.assertIn('#0ea5e9',colors);self.assertIn('#22c55e',colors)
  v.pan_pixels[0]=100;self.assertEqual(self.v.pan_pixels[0],0)
  v.set_frame(0);self.assertEqual(self.v.frame,1)
 def test_rotation_zoom(self):
  old=self.v.camera.copy();self.v._on_mouse_down(SimpleNamespace(state=0,x=0,y=0));self.v._on_mouse_drag(SimpleNamespace(x=20,y=10))
  self.assertFalse(np.allclose(old,self.v.camera));self.v._on_mouse_wheel(SimpleNamespace(delta=1));self.assertGreater(self.v.zoom,1)
if __name__=='__main__':unittest.main(verbosity=2)
