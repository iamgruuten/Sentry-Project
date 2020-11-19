//  Loading models
Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('models')
])
.then(uploadImage)

// Function for face detect and image upload
function uploadImage(){
  const con=document.querySelector('.container');
  const input=document.querySelector('#myImg');
  const imgFile=document.querySelector('#myFile');
  var can;
  var img;
  imgFile.addEventListener('change',async()=>{
    if (can) {can.remove();}
    if (img) {img.remove();}
    // Creating a html element from a blob
    img = await faceapi.bufferToImage(myFile.files[0]);
    input.src=img.src;
    const results = await faceapi.detectAllFaces(input).withFaceLandmarks()
    .withFaceDescriptors();
     console.log(results);
     // Check if array is empty to determine if image contains human--line 24
     if (Array.isArray(results) && results.length)
     {
      document.getElementById("output").innerHTML = "Human";
     }
     else
     {
      document.getElementById("output").innerHTML = "Not human";
     }  
     const faceMatcher = new faceapi.FaceMatcher(results);
     results.forEach(fd=>{
       const bestMatch = faceMatcher.findBestMatch(fd.descriptor);
       console.log(bestMatch);
     })
     // Create a canvas
     can=faceapi.createCanvasFromMedia(input);
     con.append(can);
     faceapi.matchDimensions(can,{width:input.width,height:input.height})
     // Resize the box
     const detectionsForSize = faceapi.resizeResults(results,{width:input.width,height:input.height})
     const box = results[0].detection.box;
     const facebox = new faceapi.draw.DrawBox(box);
     faceapi.draw.drawDetections(can,detectionsForSize);
  })
}

/*
prev version

const video = document.getElementById('video')

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('models'),
  faceapi.nets.faceExpressionNet.loadFromUri('models')
]).then(startVideo)

function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    stream => video.srcObject = stream,
    err => console.error(err)
  )
}

video.addEventListener('play', () => {
  const canvas = faceapi.createCanvasFromMedia(video)
  document.body.append(canvas)
  const displaySize = { width: video.width, height: video.height }
  faceapi.matchDimensions(canvas, displaySize)
  setInterval(async () => {
    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions()
    const resizedDetections = faceapi.resizeResults(detections, displaySize)
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    faceapi.draw.drawDetections(canvas, resizedDetections)
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
    faceapi.draw.drawFaceExpressions(canvas, resizedDetections)
  }, 100)
})
*/