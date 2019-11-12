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
    <div id="paint">
        <canvas id="myCanvas" width="275" height="275" "border:1px solid"></canvas>
        <button onclick="savePicture()">Click me</button>
    </div>
    <script>
var canvas = document.getElementById('myCanvas');
var ctx = canvas.getContext('2d');
ctx.fillStyle = getHexValue(255,255,255);
var painting = document.getElementById('paint');
var paint_style = getComputedStyle(painting);
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
grid_lines(5,55);
ctx.lineWidth = 3;
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
//ctx.strokeStyle = '#000000';
 
canvas.addEventListener('mousedown', function(e) {
    ctx.beginPath();
    ctx.moveTo(mouse.x, mouse.y);
 
    canvas.addEventListener('mousemove', onPaint, false);
}, false);
 
canvas.addEventListener('mouseup', function() {
    canvas.removeEventListener('mousemove', onPaint, false);
}, false);
 
var onPaint = function() {
    ctx.lineTo(mouse.x, mouse.y);
    ctx.stroke();
};
function grid_lines(side_count,grid_size)
{
   var line_count=side_count+1
   var c=document.getElementById("myCanvas")
   var ctx=c.getContext("2d")
   var w=c.width
   var h=c.height
   //alert(w+" "+h)
   var x1=0;y1=0;x2=0;y2=0;
   // left to right lines
   y1=0;y2=h;
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
}
function rgbToHex(rgb) { 
  var hex = Number(rgb).toString(16);
  if (hex.length < 2) {
       hex = "0" + hex;
  }
  return hex;
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
    //IPython.notebook.kernel.execute("tnr="+imgData+";output_data(tnr)")
    IPython.notebook.kernel.execute("tnr="+imgData+";output_data(tnr)")
}
function getHexValue(r,g,b) {   
  var red = rgbToHex(r);
  var green = rgbToHex(g);
  var blue = rgbToHex(b);
  return "#"+red+green+blue;
};
function drawLine(x1,y1,x2,y2)
{
   var c = document.getElementById("myCanvas");
   var ctx = c.getContext("2d");
   ctx.lineWidth=2;
   ctx.strokeStyle=getHexValue(120,120,120);
   ctx.beginPath();
   ctx.moveTo(x1, y1);
   ctx.lineTo(x2, y2);
   ctx.stroke();
   ctx.strokeStyle=getHexValue(0,0,0);
};

    </script>
  </body>
</html>