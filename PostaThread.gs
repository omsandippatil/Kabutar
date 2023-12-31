function doGet(e) {
  var uid = e.parameter.uid;
  var threadText = e.parameter.threadText;
  addThreadToSheet(uid, threadText);
  return ContentService.createTextOutput("successfull");
}

function addThreadToSheet(uid, threadText) {
  // Open the spreadsheet by its ID
  var spreadsheet = SpreadsheetApp.openById('Your_Spreadsheet_ID');
  var sheet = spreadsheet.getSheetByName('Sheet_Name');
  var lastRow = sheet.getLastRow() + 1;
  sheet.getRange(lastRow, 1).setValue(uid);
  sheet.getRange(lastRow, 2).setValue(lastRow - 1);
  sheet.getRange(lastRow, 3).setValue(threadText);
}
