
### Create IAM policy to deploy Mage

```
aws iam create-policy --policy-name TerraformApplyDeployMage --policy-document \
  "$(curl -s https://raw.githubusercontent.com/mage-ai/mage-ai-terraform-templates/master/aws/policies/TerraformApplyDeployMage.json)"

```
### Create AWS policies to delete resources
```
aws iam create-policy --policy-name TerraformDestroyDeleteResources --policy-document \
  "$(curl -s https://raw.githubusercontent.com/mage-ai/mage-ai-terraform-templates/master/aws/policies/TerraformDestroyDeleteResources.json)"

```

### Create IAM user
```
aws iam create-user --user-name MageDeployer
```

### Attach Policies to User

```
aws iam attach-user-policy \
  --policy-arn $(aws iam list-policies \
  --query "Policies[?PolicyName==\`TerraformApplyDeployMage\`].Arn" \
  --output text) \
  --user-name MageDeployer

```


```
aws iam attach-user-policy \
  --policy-arn $(aws iam list-policies \
  --query "Policies[?PolicyName==\`TerraformDestroyDeleteResources\`].Arn" \
  --output text) \
  --user-name MageDeployer

```

### Create Access Key for the User

```
aws iam create-access-key \
  --user-name MageDeployer \
  --output json | jq -r '"[mage-deployer]\naws_access_key_id = \(.AccessKey.AccessKeyId)\naws_secret_access_key = \(.AccessKey.SecretAccessKey)"' >> ~/.aws/credentials
export AWS_PROFILE="mage-deployer"

```