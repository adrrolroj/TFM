window.onload = () => connect();
var year_start = '2015';
var month_start = '02';
var day_start = '14';
var year_end = '2022';
var month_end = '02';
var day_end = '14';
var hospital = '4';


connect = function() {

  // Get references to elements on the page.
  //var form = document.getElementById('message-form');
  //var messageField = document.getElementById('message');
  //var messagesList = document.getElementById('messages');
  //var socketStatus = document.getElementById('status');
  //var closeBtn = document.getElementById('close');
  var arrayConnections = [
    "[\"{\\\"msg\\\":\\\"connect\\\",\\\"version\\\":\\\"1\\\",\\\"support\\\":[\\\"1\\\",\\\"pre2\\\",\\\"pre1\\\"]}\"]",
    "[\"{\\\"msg\\\":\\\"sub\\\",\\\"id\\\":\\\"TcAtwX89rKhgG6LQo\\\",\\\"name\\\":\\\"meteor.loginServiceConfiguration\\\",\\\"params\\\":[]}\"]",
    "[\"{\\\"msg\\\":\\\"sub\\\",\\\"id\\\":\\\"65qZKBe8CjpxJkiK4\\\",\\\"name\\\":\\\"polenes2015\\\",\\\"params\\\":[{\\\"selector\\\":{\\\"$and\\\":[{\\\"fecha\\\":{\\\"$gte\\\":"+year_start+month_start+day_start+",\\\"$lte\\\":"+year_end+month_end+day_end+"}},{\\\"idEstacion\\\":"+hospital+"}]},\\\"options\\\":{\\\"sort\\\":{\\\"fecha\\\":1}},\\\"jump\\\":1}]}\"]",

  ]

  print(arrayConnections)
  // Create a new WebSocket.
  var socket = new WebSocket('wss://www.polenes.com/sockjs/045/4ec9at4_/websocket');

  var messageTotal = "";

  // Handle any errors that occur.
  socket.onerror = function(error) {
    console.log('WebSocket Error: ' + error);
  };


  // Show a connected message when the WebSocket is opened.
  socket.onopen = function(event) {
    socketStatus.innerHTML = 'Connected to: ' + event.currentTarget.url;
    socketStatus.className = 'open';
    test();

  };


  // Handle messages sent by the server.
  socket.onmessage = function(event) {
    var message = event.data;
    messagesList.innerHTML += '<li class="received"><span>Received:</span>' + message + '</li>';
    messageTotal += message+"\n";

  };


  // Show a disconnected message when the WebSocket is closed.
  socket.onclose = (event) => {
    socketStatus.innerHTML = 'Disconnected from WebSocket.';
    socketStatus.className = 'closed';
    console.log('Cerrado!!')
    document.getElementById('link').download = this.hospital+"_"+this.year_start+this.month_start+this.day_start+"_"+this.year_end+this.month_end+this.day_end;

    document.getElementById('link').click();
    this.year = this.year+1;

    if(this.year<2021){
      setTimeout(function() {
        connect();
      }, 1000);
    }
  };



  let test = () => {
    // e.preventDefault();
    console.log(this.year)
    // Retrieve the message from the textarea.
    var message = messageField.value;

    arrayConnections.forEach((request) => {

      message = request;

      socket.send(message);

    });
    // // Send the message through the WebSocket.
    // message= "[\"{\\\"msg\\\":\\\"connect\\\",\\\"version\\\":\\\"1\\\",\\\"support\\\":[\\\"1\\\",\\\"pre2\\\",\\\"pre1\\\"]}\"]"
    //
    //
    // // Add the message to the messages list.
    // messagesList.innerHTML += '<li class="sent"><span>Sent:</span>' + message + '</li>';
    //
    // message = "[\"{\\\"msg\\\":\\\"sub\\\",\\\"id\\\":\\\"TcAtwX89rKhgG6LQo\\\",\\\"name\\\":\\\"meteor.loginServiceConfiguration\\\",\\\"params\\\":[]}\"]"
    //
    // socket.send(message);
    //
    // // Add the message to the messages list.
    // messagesList.innerHTML += '<li class="sent"><span>Sent:</span>' + message + '</li>';
    //
    // // Clear out the message field.
    // messageField.value = '';
    return false;

  };


  // Close the WebSocket connection when the close button is clicked.
  closeBtn.onclick = function(e) {
    e.preventDefault();

    // Close the WebSocket.
    socket.close();

    return false;
  };

  document.getElementById('link').onclick = function()
  {
    this.href = 'data:text/plain;charset=utf-11,' + encodeURIComponent(messageTotal);
  };


};
print('heu')
connect();