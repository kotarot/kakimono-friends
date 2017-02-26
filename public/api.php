<?php

$inputdata = isset($_GET['inputdata']) ? $_GET['inputdata'] : '';
$inputarray = explode(',', $inputdata);
$queryarray = array();

// Validation for inputdata
$valid = true;
foreach ($inputarray as $d) {
    $dn = (int)$d;
    if ($dn < 0 || 255 < $dn) {
        $valid = false;
        break;
    }
    $queryarray[] = (string)$dn;
}
if ($valid) {
    if (count($queryarray) != 28 * 28) {
        $valid = false;
    }
}

if ($valid) {
    // Post data
    $postdata = array(
        'inputdata' => implode(',', $queryarray)
    );
    $postdata = http_build_query($postdata, '', '&');
    $header = array(
        'Content-Type: application/x-www-form-urlencoded',
        'Content-Length: ' . strlen($postdata)
    );
    $context = array(
        'http' => array(
            'method' => 'POST',
            'header' => implode("\r\n", $header),
            'content' => $postdata
        )
    );
    echo file_get_contents('http://127.0.0.1:5000/', false, stream_context_create($context));
} else {
    echo "-1\n";
}
