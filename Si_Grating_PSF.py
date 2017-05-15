import matplotlib
matplotlib.use('TKAgg')

import FileDialog
from scipy.signal import fftconvolve, convolve2d
from scipy import fftpack
from scipy.ndimage.filters import gaussian_filter

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
import pyfits as fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import ttk
import sys
import os
import tkMessageBox

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk

from tkFileDialog import askopenfilename

# Create a new Matplotlib toolbar that does not display live x,y coordinates
# (this caused the toolbar to move around in its tk grid cell)
class my_toolbar(NavigationToolbar2TkAgg):
    def set_message(self, msg):
        pass

# Class for window to load ZYGO .xyz file and set beam location.
class LoadWindow(tk.Frame):
    counter = 0
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        # Assign tk root to an instance variable
        self.master = master

        #Set window title
        self.master.wm_title('Load Window')

        # This overrides the command to exit a tkinter window using the red exit button on mac
        # and calls a custon quit command. Without this line, the embedded matplotlib canvas
        # would cause a fatal error in python
        self.master.protocol('WM_DELETE_WINDOW', self._quit)

        # Boolean that records whether or not a zygo image has been loaded
        self.load_zygo = False

        # Set default pixel scale value and beam radius
        self.default_scale = 0.1115 #mm/pixel
        self.default_beam_r = 12.5
        self.beam_center = (17.28,45.0)
        self.current_pix_scale = 0.1115


        # Generate a figure for the load page.  
        self.beam_fig = plt.figure(figsize=(4.5,6))  

        # Use gridspec to set size ratio for main plot to side plots.
        self.beam_gs = gridspec.GridSpec(2, 2,  
                       width_ratios=[1,2],
                       height_ratios=[4,1])

        # Define the main plot and turn off axis tick labels, because they will
        # be taken care of by the side plots.
        self.beam_ax = plt.subplot(self.beam_gs[1])
        self.beam_ax.get_xaxis().set_visible(False)
        self.beam_ax.get_yaxis().set_visible(False)
        
        #Define the vertical cross-cut plot to the right of the main plot.
        self.vertical_cut_ax = plt.subplot(self.beam_gs[0], sharey = self.beam_ax)
        self.vertical_cut_ax.get_xaxis().set_visible(False)

        #Define the horizontal cross-cut plot below the main plot.
        self.horizontal_cut_ax = plt.subplot(self.beam_gs[3], sharex=self.beam_ax)
        self.horizontal_cut_ax.get_yaxis().set_visible(False)

        # Tight layout ensures that plots will fill the space defined by figsize
        # when the figure was originally generated
        self.beam_gs.tight_layout(self.beam_fig)

        # Draw the plots.
        self.beam_fig.canvas.draw()

        #Create circle artist that will not be drawn until a ZYGO image is loaded
        self.beam_region = plt.Circle(self.beam_center, self.default_beam_r, color = 'black', 
            fill=False, zorder=2)

        #Create canvas and place tk matplotlib widget in frame
        self.beam_canvas = FigureCanvasTkAgg(self.beam_fig, master=self.master)
        self.beam_canvas.get_tk_widget().grid(row=1,column=0, columnspan=4, rowspan=22, \
            sticky=('N','S','E','W'))

        #Create separate frame for toolbar (because it calls 'pack' internally and will not
        #work in gridspace)
        self.toolbar_frame = tk.Frame(self.master)
        self.toolbar_frame.grid(row=0,column=0, columnspan=7)
        self.beam_toolbar = my_toolbar(self.beam_canvas, self.toolbar_frame)
        self.beam_toolbar.update()

        #Connect mouse and keyboard to beam canvas.
        self.beam_canvas.mpl_connect('button_press_event', self.beam_click)

        #Bind arrow keys to program.
        self.master.bind('<Up>', self.key_press)
        self.master.bind('<Down>', self.key_press)
        self.master.bind('<Right>', self.key_press)
        self.master.bind('<Left>', self.key_press)

        # Quit button
        self.quit_button = tk.Button(self.master, text='Quit', command = self._quit)
        self.quit_button.grid(row=23, column=0)

        # Label to dispaly PV measurement
        self.pv_label = tk.Label(self.master, text = 'PV =')
        self.pv_label.grid(row = '6', column=5, columnspan=2, sticky='SW')
        self.pv_label.configure(font=('TkDefaultFont', 18))
        self.pv = 0.0

        # Label to display current file loaded
        self.file_label = tk.Label(self.master, text = '')
        self.file_label.grid(row = '5', column=5, columnspan=2, sticky='SWE', padx=(0,50))   
        self.file_label.configure(borderwidth=2, relief='groove')

        # Button to set pixel scale
        self.scale_button = tk.Button(self.master, text='Update Pixel Scale', \
            command=self.update_scale)
        self.scale_button.grid(row = 1, column=5, columnspan=2, sticky = 'SW')

        # Create pixel scale entry
        self.pixel_scale = tk.StringVar()
        self.pixel_scale_entry = tk.Entry(self.master, width=7, textvariable=self.pixel_scale)
        self.pixel_scale_entry.grid(row=2, column=5, sticky='N')
        self.pixel_scale_entry.insert(0, str(self.default_scale))
        tk.Label(self.master,text='mm/pix').grid(row=2, column=6, sticky='NW', padx=(0,50))

        # Button to set beam size
        self.beam_button = tk.Button(self.master, text='Update Beam Diameter', \
            command=self.update_beam)
        self.beam_button.grid(row=3, column=5, columnspan=2, sticky = 'SW')

        # Create beam size entry 
        self.beam_diameter = tk.StringVar()
        self.beam_diameter_entry = tk.Entry(self.master, width=7, textvariable=self.beam_diameter)
        self.beam_diameter_entry.grid(row=4,column=5, sticky='N')
        self.beam_diameter_entry.insert(0, str(self.default_beam_r*2.0))
        tk.Label(self.master, text='mm').grid(row=4, column=6, sticky='NW')

        # Button to browse and load xyz file
        self.load_button = tk.Button(self.master, text='Load Zygo .xyz', command=self.load_xyz)
        self.load_button.grid(row=23,column=2, sticky = 'E')

        # Button to continue to the fourier transform widget after selecting beam
        self.next_button = tk.Button(self.master, text="Next", 
                                command=self.open_psf_window)
        self.next_button.grid(row=23,column=6, sticky = 'W')


    # Function for loading zygo xyz files and converting them into a numpy array 
    # compatable with pyplot function 'imshow'
    def load_xyz(self):

        # Browse files in order to select .xyz file to load
        self.load_filename = askopenfilename()

        # Make sure file contains '.xyz' extension 
        if '.xyz' in self.load_filename:
            self.current_file = self.load_filename.split('/')[-1]
            pass
        elif self.load_filename == '':
            return
        else:
            tkMessageBox.showwarning(\
                'Warning', 'Unrecognized extension. Please choose a Zygo .xyz file.')
            #print 'Unrecognized extension. Please choose a Zygo .xyz file to load.'
            return

        # Open specified xyz file and get all lines in file
        read_xyz = open(self.load_filename, 'r')
        xyz_lines = read_xyz.readlines()

        # Create a temporary text file to write to 
        temp_write = open('temp_file.txt', 'w')

        # Loop that will skip over .xyz file header and replace columns containing
        # 'No Data' with 'nan'
        i=0
        for line in xyz_lines:
            line=line.replace('No Data', 'nan')
            if i>=14:
                temp_write.write(line)
            i+=1

        #Close open text files
        read_xyz.close()
        temp_write.close()

        # Use numpy.loadtxt to generate a numpy array from ZYGO data that was 
        # made into a numpy readable text file. 
        zygo_read_x, zygo_read_y, zygo_read_z = np.loadtxt('temp_file.txt', \
            unpack=True, usecols=[0,1,2])

        #Set x,y grid so that they start at 0
        zygo_read_x = zygo_read_x.astype(int) - int(min(zygo_read_x))
        zygo_read_y = zygo_read_y.astype(int) - int(min(zygo_read_y))

        # Initialize array to store zygo image data
        self.zygo_img = np.empty((max(zygo_read_x)+1, max(zygo_read_y)+1))
        
        # Populate zygo_img array with data from temporary text file
        for j in range(0,len(zygo_read_z)):
            self.zygo_img[zygo_read_x[j], zygo_read_y[j]] = zygo_read_z[j]

        # Once 'zygo_img' array is successfully populated we can use the update_plots()
        # function to write everything to the embedded matplotlib canvas
        self.update_plots()
        self.load_zygo = True

        #Update PV value in label
        self.update_pv()

        #Update filename displayed in gui to current loaded file
        self.update_filename()

    def load_zygo_test_mode(self):
        self.zygo_img = fits.getdata('test_I02.fits')

        # Once 'zygo_img' array is successfully populated we can use the update_plots()
        # function to write everything to the embedded matplotlib canvas
        self.update_plots()
        self.load_zygo = True

        #Update PV value in label
        self.update_pv()

        #Update filename displayed in gui to current loaded file
        self.update_filename()


    def update_scale(self):
        if self.load_zygo==False:
            print 'Please load an image before altering the pixel scale.'
        else:
            try:
                self.current_pix_scale = float(self.pixel_scale.get())
                self.update_plots()
            except ValueError:
                print 'Warning: Could not convert string to float. Scale not updated.'


    def update_beam(self):
        if self.load_zygo==False:
            print 'Please load zygo image before updating beam size.'
        else:
            try:
                self.beam_region.radius = float(self.beam_diameter_entry.get())/2.0
                self.update_plots()
            except ValueError:
                print 'Warning: Could not convert string to float. Beam size not updated.'


    def beam_click(self, event):
        if event.inaxes==self.beam_ax and self.load_zygo==False:
            print 'Please load zygo image in order to place beam.'
        elif event.inaxes==self.beam_ax and self.load_zygo==True:
            self.beam_center = (event.xdata, event.ydata)
            self.update_plots()

        self.master.focus()


    def key_press(self, event):

        if self.load_zygo == False:
             print 'Please load zygo image in order to place beam.'
        elif (event.keysym=='Up' or event.keysym=='Down' or \
            event.keysym == 'Left' or event.keysym == 'Right'):

            try:
                temp_pix_scale = float(self.pixel_scale.get())
            except ValueError:
                print 'Could not convert string to float. Need valid pixel scale to move beam.'
                return

            if event.keysym=='Up':
                self.beam_center= (self.beam_center[0], self.beam_center[1]+(5.0*temp_pix_scale))
            elif event.keysym=='Down':
                self.beam_center= (self.beam_center[0], self.beam_center[1]-(5.0*temp_pix_scale))
            elif event.keysym=='Left':
                self.beam_center= (self.beam_center[0]-(5.0*temp_pix_scale), self.beam_center[1])
            elif event.keysym=='Right':
                self.beam_center= (self.beam_center[0]+(5.0*temp_pix_scale), self.beam_center[1])

            self.update_plots()


    def update_plots(self):
        self.vertical_cut_ax.cla()
        self.horizontal_cut_ax.cla()
        self.beam_ax.cla()

        self.beam_ax.imshow(np.rot90(self.zygo_img), \
            extent=[0,self.zygo_img.shape[0]*self.current_pix_scale,\
            0,self.zygo_img.shape[1]*self.current_pix_scale], zorder=1)

        self.beam_ax.add_artist(self.beam_region)
        self.beam_cross, = self.beam_ax.plot(self.beam_center[0], self.beam_center[1],\
            '+', ms = 8, color = 'black', zorder=2)

        self.zygo_x_arr = np.linspace(0, self.zygo_img.shape[0]*self.current_pix_scale, \
            self.zygo_img.shape[0])
        self.zygo_y_arr = np.linspace(0, self.zygo_img.shape[1]*self.current_pix_scale, \
            self.zygo_img.shape[1])

        self.beam_center_x_index = np.argmin(np.abs(self.zygo_x_arr-self.beam_center[0]))
        self.beam_center_y_index = np.argmin(np.abs(self.zygo_y_arr-self.beam_center[1]))
        self.vertical_crosscut = self.zygo_img[self.beam_center_x_index,:]
        self.vertical_cut_ax.plot(self.vertical_crosscut, self.zygo_y_arr)
        self.horizontal_crosscut = self.zygo_img[:,self.beam_center_y_index]
        self.horizontal_cut_ax.plot(self.zygo_x_arr, self.horizontal_crosscut)

        self.beam_region.center = self.beam_center
        self.beam_cross.set_data(self.beam_center[0], self.beam_center[1])

        self.beam_ax.set_ylim(min(self.zygo_y_arr), max(self.zygo_y_arr))

        self.horizontal_cut_ax.axvline(x=(self.beam_center[0] + self.beam_region.radius),\
            color='black', linestyle='--')
        self.horizontal_cut_ax.axvline(x=(self.beam_center[0] - self.beam_region.radius),\
            color='black', linestyle='--')
        self.vertical_cut_ax.axhline(y=(self.beam_center[1] + self.beam_region.radius),\
            color='black', linestyle='--')
        self.vertical_cut_ax.axhline(y=(self.beam_center[1] - self.beam_region.radius),\
            color='black', linestyle='--')

        self.beam_fig.canvas.draw()
        self.update_pv()


    def update_pv(self):

        x_mesh, y_mesh = np.meshgrid(self.zygo_x_arr, self.zygo_y_arr[::-1])
        beam_array = np.copy(np.rot90(self.zygo_img))

        beam_array[np.where(((x_mesh-self.beam_center[0])**2 + \
            (y_mesh-self.beam_center[1])**2) >=(self.beam_region.radius)**2)] = np.nan

        self.pv = (np.nanmax(beam_array)-np.nanmin(beam_array))*633.0

        self.pv_label.config(text = 'PV = %.0f nm'%(self.pv))


    def update_filename(self):
        self.file_label.configure(text=self.current_file, borderwidth=2, relief='groove')


    def open_psf_window(self):
        if self.load_zygo:
            self.PSF_window = tk.Toplevel(self)
            self.app = PSFWindow(self.PSF_window, np.rot90(self.zygo_img), \
                self.current_pix_scale, self.beam_region.radius, self.beam_region.center)
        else:
            print 'Please load ZYGO image and place beam before proceeding.'


    def _quit(self):
        self.master.quit()
        self.master.destroy()




#New class for doing fourier transform and modelling
class PSFWindow(tk.Frame):
    def __init__(self, master, zygo_img, pix_scale, beam_radius, beam_center):
        tk.Frame.__init__(self)

        #Set instance of tk toplevel fed in by LoadWindow
        self.master = master

        #Assign variables passed from load window to instance variables
        self.zygo_img = zygo_img
        self.pix_scale = pix_scale
        self.beam_radius = beam_radius
        self.beam_center = np.asarray(beam_center)
        self.taper_sigma = 0.75 #mm

        #Set window title
        self.master.wm_title('PSF Window')

        #Establish use of grid for gui layout (as opposed to 'pack')
        self.master.grid_rowconfigure(0)
        self.master.grid_columnconfigure(0)

        # This overrides the command to exit a tkinter window using the red exit button on mac
        # and calls a custon quit command. Without this line, the embedded matplotlib canvas
        # would cause a fatal error in python
        self.master.protocol('WM_DELETE_WINDOW', self._quit)

        # Generate a figure for the load page.  
        self.psf_fig = plt.figure(figsize=(12,6.6))  

        # Use gridspec to set size ratio for main plot to side plots.
        self.psf_gs = gridspec.GridSpec(2, 3,  
                       width_ratios=[4,4,1],
                       height_ratios=[4,1])


        # Define the main plot and turn off axis tick labels, because they will
        # be taken care of by the side plots.
        self.beam_ax = plt.subplot(self.psf_gs[0])
        self.beam_ax.get_xaxis().set_visible(False)
        self.beam_ax.get_yaxis().set_visible(False)

        self.psf_ax = plt.subplot(self.psf_gs[1])
        self.psf_ax.get_xaxis().set_visible(False)
        self.psf_ax.get_yaxis().set_visible(False)

        #Define the vertical cross-cut plot to the right of the main plot.
        self.psf_vcut_ax = plt.subplot(self.psf_gs[2], sharey = self.psf_ax)
        self.psf_vcut_ax.get_xaxis().set_visible(False)
        self.psf_vcut_ax.yaxis.tick_right()

        #Define the horizontal cross-cut plot below the main plot.
        self.psf_hcut_ax = plt.subplot(self.psf_gs[4], sharex=self.psf_ax)
        self.psf_hcut_ax.get_yaxis().set_visible(False)

        self.psf_gs.tight_layout(self.psf_fig)

        self.psf_fig.canvas.draw()

        self.psf_canvas = FigureCanvasTkAgg(self.psf_fig, master=self.master)
        self.psf_canvas.get_tk_widget().grid(row=0,column=0, columnspan=30, rowspan=22, \
            sticky=('N','S','E','W'))

        #Format zygo image for analysis
        self.format_image()

        #Apply gaussian taper to image
        self.edge_taper()

        #Spatial resolution input
        self.pix_scale_input = tk.StringVar()
        self.pix_scale_entry = tk.Entry(self.master, width=7, textvariable=self.pix_scale_input)
        self.pix_scale_entry.grid(row=20,column=0, sticky = 'E')
        self.pix_scale_entry.insert(0, str(self.pix_scale))
        tk.Label(self.master, text='mm/pix').grid(row=20, column=1, sticky='W')


    def format_image(self):

        self.img_grid_size = 1024
        self.zygo_format_img = np.empty((self.img_grid_size,self.img_grid_size))
        self.zygo_format_img.fill(np.nan)

        #Define region on new grid for zygo image to be pasted onto
        x_min = int((self.img_grid_size/2.0) - self.beam_center[0]/self.pix_scale)
        y_min = int((self.img_grid_size/2.0) - self.beam_center[1]/self.pix_scale)

        x_max = x_min + int(self.zygo_img.shape[1])
        y_max = y_min + int(self.zygo_img.shape[0])


        top_cut = int(self.zygo_img.shape[0])
        bottom_cut = int(0)
        right_cut = int(self.zygo_img.shape[1])
        left_cut = int(0)
        
        if y_max>self.img_grid_size:
            top_cut = int(self.zygo_img.shape[0] - (y_max-self.img_grid_size))
            y_max = self.img_grid_size
        if y_min<0: 
            bottom_cut = np.absolute(y_min)
            y_min = 0



        #if x_max>self.img.grid_size:

        
        #self.psf_fig.canvas.draw()

        #Place zygo image onto new grid 
        self.zygo_format_img[y_min:y_max, x_min:x_max] = self.zygo_img[bottom_cut:top_cut, left_cut:right_cut]


        self.beam_ax.imshow(self.zygo_format_img)
        #Create xmesh, ymesh and rmesh arrays
        self.zygo_ax_array = np.arange(0, self.img_grid_size)*self.pix_scale
        self.x_mesh, self.y_mesh = np.meshgrid(self.zygo_ax_array, self.zygo_ax_array)
        self.r_mesh = np.sqrt((self.x_mesh-(self.img_grid_size/2.0)*self.pix_scale)**2 \
            + (self.y_mesh-(self.img_grid_size/2.0)*self.pix_scale)**2)


        #Eliminate nan values. Need to find a less stupid way to do this in the future
        zygo_nan_copy = np.copy(self.zygo_format_img)
        self.zygo_format_img[np.where(np.isnan(self.zygo_format_img)==True)] = 0.0
        #self.psf_ax.imshow(self.zygo_format_img)
        #self.psf_fig.canvas.draw()

        #Blur image
        self.blurred_img = gaussian_filter(self.zygo_format_img, sigma=10)
        self.blurred_img[np.where(self.r_mesh<=(self.beam_radius*1.05))] = \
            self.zygo_format_img[np.where(self.r_mesh<=(self.beam_radius*1.05))]
        self.zygo_format_img = np.copy(self.blurred_img)




    def edge_taper(self):

        taper_radius = self.beam_radius
        
        self.taper_matrix = np.exp(-(((self.r_mesh-taper_radius))**2)/(2.0*(self.taper_sigma**2)))
        self.taper_matrix[np.where(self.r_mesh<=taper_radius)]=1.0


        self.zygo_format_img*=self.taper_matrix
        self.zygo_format_img+=1.0
        self.psf = (np.absolute(fftpack.fftshift(fftpack.fft2(self.zygo_format_img))))

        self.psf/=np.nanmax(self.psf)
        self.psf_ax.imshow(np.log10(np.sqrt(self.psf)),clim = (-3.5, 0.0), aspect=1.0) 
        #self.beam_ax.imshow(self.taper_matrix)
        self.psf_gs.tight_layout(self.psf_fig)

        self.beam_ax.set_xlim(100,900)
        self.beam_ax.set_ylim(100,900)

        #self.psf_ax.set_xlim(0,850)
        #self.psf_ax.set_ylim(150,850)

        #self.vertical_crosscut.set_ylim(100,900)
        #self.horizontal_crosscut.set_xlim(100,900)

        self.psf_fig.canvas.draw()


        print np.mean(self.zygo_format_img)

  

    def _quit(self):
        self.master.destroy()


#Test commit

root = tk.Tk()
root.lift()
root.attributes('-topmost',True)
root.after_idle(root.attributes,'-topmost',False)
main = LoadWindow(root)
main.grid(row=0,column=0)
root.mainloop()