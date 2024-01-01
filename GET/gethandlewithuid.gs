function getHandlesFromThreads() {
  
  var userSpreadsheetId = 'User_Spreadsheet_ID';
  var threadsSpreadsheetId = 'User_Threadsheet_ID';
  var userSheetName = 'user';
  var threadsSheetName = 'threads';
  
  var userSpreadsheet = SpreadsheetApp.openById(userSpreadsheetId);
  var threadsSpreadsheet = SpreadsheetApp.openById(threadsSpreadsheetId);
  
  var userSheet = userSpreadsheet.getSheetByName(userSheetName);
  var threadsSheet = threadsSpreadsheet.getSheetByName(threadsSheetName);

  var userDataRange = userSheet.getDataRange();
  var userValues = userDataRange.getValues();

  var threadsDataRange = threadsSheet.getDataRange();
  var threadsValues = threadsDataRange.getValues();

  var userHeaders = userValues[0];
  var userUidIndex = userHeaders.indexOf('uid');
  var userHandleIndex = userHeaders.indexOf('handle');

  var threadsHeaders = threadsValues[0];
  var threadsUidIndex = threadsHeaders.indexOf('uid');

  var uidToHandleMap = {};
  for (var i = 1; i < userValues.length; i++) {
    var userRow = userValues[i];
    var uid = userRow[userUidIndex];
    var handle = userRow[userHandleIndex];
    uidToHandleMap[uid] = handle;
  }

  for (var j = 1; j < threadsValues.length; j++) {
    var threadsRow = threadsValues[j];
    var uidFromThreads = threadsRow[threadsUidIndex];

    if (uidToHandleMap.hasOwnProperty(uidFromThreads)) {
      var handle = uidToHandleMap[uidFromThreads];
      Logger.log('UID: ' + uidFromThreads + ', Handle: ' + handle);
    } else {
      Logger.log('UID: ' + uidFromThreads + ', Handle not found');
    }
  }
}
