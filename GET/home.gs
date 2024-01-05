function getThreadAndLikeDataFromBottomToTop() {
  var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  var handlesSheet = spreadsheet.getSheetByName('user');
  var threadsSheet = spreadsheet.getSheetByName('threads');
  var handlesData = handlesSheet.getDataRange().getValues();
  
  var uidHandleMap = {};
  handlesData.forEach(function (row) {
    var uid = row[0];
    var handle = row[6];
    uidHandleMap[uid] = handle;
  });

  var threadsData = threadsSheet.getDataRange().getValues();

  for (var i = threadsData.length - 1; i >= 1; i--) {
    var row = threadsData[i];
    var thread = row[2];
    var likes = row[3];
    var uid = row[0];

    if (uidHandleMap[uid]) {
      var handle = uidHandleMap[uid];
      Logger.log('Handle: ' + handle + ', Thread: ' + thread + ', Likes: ' + likes);
    }
  }
}
