function doGet(e) {
  var followerUid = e.parameter.followerUid;
  var followingUid = e.parameter.followingUid;

  if (followerUid && followingUid) {
    addFollowToSheet(followerUid, followingUid);
    return ContentService.createTextOutput("successful");
  } else {
    return ContentService.createTextOutput("failed");
  }
}

function addFollowToSheet(followerUid, followingUid) {
  var sheetName = "follow";
  var sheet = getSheet(sheetName);

  var newRow = [followerUid, followingUid]; // Assuming you want to timestamp the follow

  sheet.appendRow(newRow);
}

function getSheet(sheetName) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName(sheetName);

  if (!sheet) {
    sheet = ss.insertSheet(sheetName);
    // If the sheet doesn't exist, you may want to add headers
    sheet.appendRow(["Follower UID", "Following UID"]);
  }

  return sheet;
}
