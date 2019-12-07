<!DOCTYPE HTML>
<html>
  <head>
    <style>
      body {
        margin: 0px;
        padding: 0px;
      }
    </style>
  </head>
  <body>
  <style type="text/css">
.divLeft {
    width:300px;
    display:block;
    float: left;
}
.divRight {
    width:200px;
    display:block;
    float:left
}
</style>
<span class="divLeft"/><canvas id="myCanvas" width="?width?" height="?height?" "border:1px solid"></canvas>
<span class="divRight"><table style="width:80%">
  <tr>
    <th>Class</th><th>Count</th>
  </tr>
  <tr>
    <td>0</td><td>?class00?</td>
  </tr>
  <tr>
    <td>1</td><td>?class01?</td>
  </tr>
  <tr>
    <td>2</td><td>?class02?</td>
  </tr>
  <tr>
    <td>3</td><td>?class03?</td>
  </tr>
  <tr>
    <td>4</td><td>?class04?</td>
  </tr>
  <tr>
    <td>5</td><td>?class05?</td>
  </tr>
  <tr>
    <td>6</td><td>?class06?</td>
  </tr>
  <tr>
    <td>7</td><td>?class07?</td>
  </tr>
  <tr>
    <td>8</td><td>?class08?</td>
  </tr>
  <tr>
    <td>9</td><td>?class09?</td>
  </tr>
</table></span>
<div>
    <button onclick="savePicture()">Save</button>
    <button onclick="do_erase_toggle()">Erase</button>
    <button onclick="clear_canvas()">Clear</button>
    <button onclick="redrawGrid()">Grid</button>
    <select id='theSelect' onchange="?class_label_choice?">
       <option value="0"  >Class 0</option>
       <option value="1"  >Class 1</option>
       <option value="2"  >Class 2</option>
       <option value="3"  >Class 3</option>
       <option value="4"  >Class 4</option>
       <option value="5"  >Class 5</option>
       <option value="6"  >Class 6</option>
       <option value="7"  >Class 7</option>
       <option value="8"  >Class 8</option>
       <option value="9"  >Class 9</option>
    </select>
</div>    
    <script>
var canvas = document.getElementById('myCanvas');
var ctx = canvas.getContext('2d');
ctx.fillStyle = getHexValue(255,255,255);
var erase_toggle=0
var element=document.getElementById('theSelect')
element.value="?selval?"
//var painting = document.getElementById('paint');
//var paint_style = getComputedStyle(painting);
//tw = parseInt(paint_style.getPropertyValue('width'));
//canvas.height = parseInt(paint_style.getPropertyValue('height'));
ctx.fillRect(0, 0,canvas.width,canvas.height);

var mouse = {x: 0, y: 0};

canvas.addEventListener('mousemove', function(e) {
  mouse.x = e.pageX - this.getBoundingClientRect().left;
  mouse.y = e.pageY - this.getBoundingClientRect().top;
  //mouse.x = e.pageX - this.offsetLeft;
  //mouse.y = e.pageY - this.offsetTop;
}, false);
grid_lines(?grid_count?,?pxl?);
ctx.lineWidth = ?line_wd?;
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
//ctx.strokeStyle = '#000000';
 
canvas.addEventListener('mousedown', function(e) {
    //ctx.closePath();
    //ctx.beginPath();
    var c=document.getElementById("myCanvas")
    var ctx=c.getContext("2d")
    ctx.beginPath();
    ctx.moveTo(mouse.x, mouse.y);
 
    canvas.addEventListener('mousemove', onPaint, false);
}, false);
 
canvas.addEventListener('mouseup', function() {
    canvas.removeEventListener('mousemove', onPaint, false);
}, false);
 
var onPaint = function() {
    //alert('1')
    var c=document.getElementById("myCanvas")
    var ctx=c.getContext("2d")
    //ctx.beginPath();
    ctx.strokeStyle=getHexValue(erase_toggle,erase_toggle,erase_toggle);
    ctx.lineWidth=?line_wd?
    ctx.lineTo(mouse.x, mouse.y);
    //ctx.closePath()
    ctx.stroke();
    //hold_up(1)
    //grid_lines(?grid_count?,?pxl?);
    ctx.moveTo(mouse.x, mouse.y);
};
function redrawGrid()
{
  grid_lines(?grid_count?,?pxl?);
}
// function hold_up(t_count)
// {
//   var i=0 
//   console.log('in hold up '+t_count)
//   var prev_time=Date.now()
//   var time_diff=0
//   var dummy=0
//   while(time_diff<1)
//   {
//       Math.log10(prev_time)
//       curr_time=Date.now()
//       time_diff=curr_time-prev_time
//   }    
//   //IPython.notebook.kernel.execute("import time;time.sleep(1)")
//}
/* function change_class_label_no_colab()
{
  var e = document.getElementById("theSelect");
  var value = e.options[e.selectedIndex].value;
  var text = e.options[e.selectedIndex].text;
  console.log(' in not colab class label '+text+" "+value)
  int_value=parseInt(value) 
  IPython.notebook.kernel.execute("tnr02="+int_value+";update_class_number(tnr02)")
  //IPython.notebook.kernel.execute("tnr02="+int_value)
} */
function grid_lines(side_count,grid_size)
{
   var line_count=side_count+1
   var c=document.getElementById("myCanvas")
   var ctx=c.getContext("2d")
   var w=c.width
   var h=c.height
   //alert(w+" "+h)
   ctx.strokeStyle=getHexValue(255,0,0);
   //ctx.beginPath();
//};
   var x1=0;y1=0;x2=0;y2=0;
   // left to right lines
   y1=0;y2=h;
   line_count=6
   for(i=0;i<line_count;i++)
   {
      x1=i*grid_size;
      x2=x1;
      drawLine(x1,y1,x2,y2);
      //ctx.moveTo(x1,y1)
      //ctx.moveTo(x2,y2)
   }
   // top to bottom lines
   x1=0;w2=w;
   for(i=0;i<line_count;i++)
   {
      y1=i*grid_size
      y2=y1;
      drawLine(x1,y1,x2,y2);
      //ctx.moveTo(x1,y1)
      //ctx.moveTo(x2,y2)
   }
   //ctx.stroke();
}
function do_erase_toggle()
{
   console.log('in do erase '+erase_toggle) 
   if(erase_toggle!=0)
   {
      erase_toggle=0
   }
   else
   {
      erase_toggle=255
      console.log('in do erase '+erase_toggle) 
   }   
}
function rgbToHex(rgb) { 
  var hex = Number(rgb).toString(16);
  if (hex.length < 2) {
       hex = "0" + hex;
  }
  return hex;
};
function clear_canvas()
{
  var c=document.getElementById("myCanvas")
  var ctx=c.getContext("2d")
  ctx.fillStyle = getHexValue(255,255,255);
  ctx.fillRect(0, 0,c.width,c.height);
  grid_lines(?grid_count?,?pxl?);
}
function change_class_label_no_colab()
{
  var e = document.getElementById("theSelect");
  var value = e.options[e.selectedIndex].value;
  var text = e.options[e.selectedIndex].text;
  console.log(' in not colab class label '+text+" "+value)
  int_value=parseInt(value) 
  IPython.notebook.kernel.execute("tnr02="+int_value+";update_class_number(tnr02)")
  //IPython.notebook.kernel.execute("tnr02="+int_value)
}
function change_class_label_colab() {
 
  var e = document.getElementById("theSelect");
  var value = e.options[e.selectedIndex].value;
  var text = e.options[e.selectedIndex].text;
  console.log(' in not colab class label '+text+" "+value)
  int_value=parseInt(value) 
  var t_ark=int_value
  console.log(' in colab label output') 
  google.colab.kernel.invokeFunction(
  'notebook.updateClassLabel', // The callback name.
  [t_ark, 'world!'], // The arguments.
  {}); // kwargs
//const text = result.data['application/json'];
//document.querySelector("#output-area").appendChild(document.createTextNode(text.result));
};
function not_colab_output(targ)
{
   console.log(' in not colab ')
   //IPython.notebook.kernel.execute("tnr="+targ) 
   IPython.notebook.kernel.execute("tnr="+targ+";output_data(tnr)")
}
function colab_output(targ) {
    t_ark=targ
    console.log(' in colab output') 
    google.colab.kernel.invokeFunction(
    'notebook.updateImageData', // The callback name.
    [t_ark, 'world!'], // The arguments.
    {}); // kwargs
  //const text = result.data['application/json'];
  //document.querySelector("#output-area").appendChild(document.createTextNode(text.result));
};
function savePicture()
{
    //myImage.src='image2.jpg';
    //context.drawImage(myImage,0,0);
    //var imgData = context.getImageData(0,0,canvasWidth,canvasHeight).data;
    var c=document.getElementById("myCanvas")
    var ctx=c.getContext("2d")
    //alert(c.width+" "+c.height)
    var imgData = ctx.getImageData(0,0,c.width,c.height).data;
    tStr = JSON.stringify(imgData)
    t_len=imgData.length
    var nw_imageData=new Array(t_len/4)
    var t_index=0
    for(var i=0;i<t_len;i=i+4)
    {
       nw_imageData[t_index]=imgData[i];
       t_index=t_index+1;
    }
    imgData=nw_imageData
    //IPython.notebook.kernel.execute("tnr="+imgData+";output_data(tnr)")
    //IPython.notebook.kernel.execute("tnr="+imgData+";output_data(tnr)")
    // template replaced with a function based on if the notebook is run in colab or mac/pc
    ?output_data?
}
function getHexValue(r,g,b) {   
  var red = rgbToHex(r);
  var green = rgbToHex(g);
  var blue = rgbToHex(b);
  return "#"+red+green+blue;
};
function drawLine(x1,y1,x2,y2)
{
   // used for debugging
   //hold_up(2)
   var c = document.getElementById("myCanvas");
   var ctx = c.getContext("2d");
   ctx.lineWidth=2;
   ctx.beginPath();
   ctx.strokeStyle=getHexValue(255,0,0);
   //hold_up(3)
   
   ctx.moveTo(x1, y1);
   ctx.lineTo(x2, y2);
   //hold_up(4)
   //ctx.stroke();
   ctx.closePath();
   ctx.stroke();
   ctx.strokeStyle=getHexValue(255,0,0);
   //hold_up(4)
};

    </script>
  </body>
</html>