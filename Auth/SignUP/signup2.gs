function doGet(e) {
  var handleToCheck = e.parameter.handle;
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  var handleColumnValues = sheet.getRange("G:G").getValues().flat();
  var ishandleAvailable = handleColumnValues.includes(handleToCheck);
  return ContentService.createTextOutput(ishandleAvailable.toString());
}
