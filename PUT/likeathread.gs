function doGet(e) {
  var uid = e.parameter.uid;
  var threadId = e.parameter.threadId;

  if (uid && threadId) {
    addDataToSheet(uid, threadId);
    return ContentService.createTextOutput('successful');
  } else {
    return ContentService.createTextOutput('Failed');
  }
}

function addDataToSheet(uid, threadId) {
  var spreadsheetId = 'Your_Sheet_ID';
  var sheetName = 'Sheet_Name';
  var sheet = SpreadsheetApp.openById(spreadsheetId).getSheetByName(sheetName);
  var lastRow = sheet.getLastRow() + 1;

  sheet.getRange(lastRow, 1).setValue(uid);
  sheet.getRange(lastRow, 2).setValue(threadId);
}
